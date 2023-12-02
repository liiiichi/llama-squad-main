import os
from dataclasses import dataclass
from typing import Optional
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import load_from_disk
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
)
from trl import SFTTrainer
from llama_squad import SquadDataCollator

@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(default=-1)
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=32768)
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    dataset_name: Optional[str] = field(default="timdettmers/openassistant-guanaco")
    use_4bit: Optional[bool] = field(default=True)
    num_train_epochs: Optional[int] = field(default=1)
    fp16: Optional[bool] = field(default=False)
    bf16: Optional[bool] = field(default=True)
    packing: Optional[bool] = field(default=False)
    gradient_checkpointing: Optional[bool] = field(default=True)
    optim: Optional[str] = field(default="paged_adamw_32bit")
    lr_scheduler_type: str = field(default="constant")
    max_steps: int = field(default=10000)
    warmup_ratio: float = field(default=0.03)
    group_by_length: bool = field(default=True)
    save_steps: int = field(default=10)
    logging_steps: int = field(default=10)
    merge_and_push: Optional[bool] = field(default=False)
    output_dir: str = field(default="./results")
    resume_from_checkpoint: Optional[str] = field(default=None)

def setup_distributed_training(args):
    # Set the device based on local_rank
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    # Initialize the distributed environment
    dist.init_process_group(backend='nccl')
    return device

def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit, bnb_4bit_quant_type=args.bnb_4bit_quant_type, bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=args.use_nested_quant,
    )
    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb_config, device_map=device_map, use_flash_attention_2=use_flash_attention)
    model.config.pretraining_tp = 1
    peft_config = LoraConfig(lora_alpha=script_args.lora_alpha, lora_dropout=script_args.lora_dropout, r=script_args.lora_r, bias="none", task_type="CAUSAL_LM")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Move the model to the specified device (GPU)
    model.to(device)

    # Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=[args.local_rank])
        
    return model, peft_config, tokenizer

def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def train_per_gpu(gpu, args):
    # Setup distributed training for this GPU
    device = setup_distributed_training(args)

    training_arguments = TrainingArguments(
        output_dir=script_args.output_dir, per_device_train_batch_size=script_args.per_device_train_batch_size, gradient_accumulation_steps=script_args.gradient_accumulation_steps, optim=script_args.optim, save_steps=script_args.save_steps, logging_steps=script_args.logging_steps, learning_rate=script_args.learning_rate, fp16=script_args.fp16, bf16=script_args.bf16, max_grad_norm=script_args.max_grad_norm, max_steps=script_args.max_steps, warmup_ratio=script_args.warmup_ratio, group_by_length=script_args.group_by_length, lr_scheduler_type=script_args.lr_scheduler_type,
    )

    # Prepare model and other training components
    model, peft_config, tokenizer = create_and_prepare_model(script_args)
    model.config.use_cache = False
    dataset = load_from_disk(script_args.dataset_name)["train"]
    train_data = prepare_dataloader(dataset, script_args.per_device_train_batch_size)
    tokenizer.padding_side = "right"
    data_collator = SquadDataCollator(tokenizer=tokenizer, mlm=False)
    trainer = SFTTrainer(
        model=model, train_dataset=dataset, peft_config=peft_config, dataset_text_field="text", max_seq_length=script_args.max_seq_length, tokenizer=tokenizer, args=training_arguments, packing=script_args.packing, data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)

    if script_args.merge_and_push:
    output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
    trainer.model.save_pretrained(output_dir)
    del model
    torch.cuda.empty_cache()
    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()
    output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)

if __name__ == "__main__":
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Check if we are using more than one GPU and launch a process for each
    if script_args.local_rank != -1:
        train_per_gpu(script_args.local_rank, script_args)
    else:
        # For single GPU or CPU training, just call the function directly
        train_per_gpu(0, script_args)
