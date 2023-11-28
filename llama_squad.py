from transformers import DataCollatorForLanguageModeling
import torch

class SquadDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
            super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
            # Assuming the tokenizer has been updated with the custom tokens
            self.answer_start_token_id = tokenizer.convert_tokens_to_ids("<answer_start>")
            self.answer_end_token_id = tokenizer.convert_tokens_to_ids("<answer_end>")

    def __call__(self, examples):
        batch = super().__call__(examples)

        # Only apply cross entropy loss to the answer part of the labels
        for idx, label in enumerate(batch["labels"]):
            # Find the positions of the start and end tokens
            answer_start_positions = (label == self.answer_start_token_id).nonzero(as_tuple=True)[0]
            answer_end_positions = (label == self.answer_end_token_id).nonzero(as_tuple=True)[0]

            answer_start = answer_start_positions[0]
            answer_end = answer_end_positions[0]
            label[:answer_start] = -100
            label[answer_end + 1:] = -2
            batch["labels"][idx] = label

        return batch
