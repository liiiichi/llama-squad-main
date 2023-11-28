import json
from dataclasses import dataclass, field
from typing import Optional

from datasets import DatasetDict, load_dataset, Dataset
from transformers import HfArgumentParser

from model import DEFAULT_SYSTEM_PROMPT, get_prompt

import random

@dataclass
class ScriptArguments:
    prompt: Optional[str] = field(
        default="single_turn",
        metadata={"help": "single_turn, multi_turn"},
    )
    dataset: Optional[str] = field(
        default="data/squad_v2",
    )

def filter_long_text_entries(dataset, max_length=115200):
    filtered_entries = {'text': [], 'context': [], 'question': [], 'answers': []}

    for entry in dataset:
        if len(entry["text"]) <= max_length:
            for key in filtered_entries.keys():
                filtered_entries[key].append(entry[key])

    return filtered_entries

def select_random_subset(dataset, num_examples=1000):
    # Randomly choose indices for the subset
    selected_indices = random.sample(range(len(dataset)), num_examples)
    # Create a subset based on these indices
    subset = dataset.select(selected_indices)
    return subset

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

def get_single_turn_prompt_and_response(item, all_answers=False):
    context = item["context"]
    question = item["question"]
    answers = item["answers"]["text"]
    if len(answers) == 0:
        answers = ["?"]
    answers = json.dumps(answers) if all_answers else f'"{answers[0]}"'

    return {
        "text": get_prompt(
            f"""\
Extract from the following context the minimal span word for word that best answers the question. Think step by step and explain your reasoning. Then give the answer in JSON format as follows:
```json
{{
  "answer": ...
}}
```
If the answer is not in the context, the answer should be "?".
Question: {question}
Context: {context}""",
            [],
            SYSTEM_PROMPT,
        )
        + f""" \
```json
{{<answer_start>
  "answer": {answers}
  <answer_end>
}}
``` </s>"""
    }


def get_multi_turn_prompt_and_response(item, all_answers=False):
    context = item["context"]
    question = item["question"]
    answers = item["answers"]["text"]
    if len(answers) == 0:
        answers = ["?"]
    answers = json.dumps(answers) if all_answers else f'"{answers[0]}"'

    return {
        "text": get_prompt(
            """\
Now give the answer in JSON format as follows:
```json
{
  "answer": ...
}
```
If the answer is not in the context, the answer should be "?".
""",
            [
                (
                    f"""\
Use the following context to answer the question. Think step by step and explain your reasoning.
Context: {context}
Question: {question}""",
                    "",
                ),
                (
                    f"""\
Extract the minimal span word for word from the context that best answers the question.
        """,
                    "",
                ),
            ],
            SYSTEM_PROMPT,
        )
        + f""" \
```json
{{
  "answer": {answers}
}}
``` </s>"""
    }


instruction = {
    "single_turn": get_single_turn_prompt_and_response,
    "multi_turn": get_multi_turn_prompt_and_response,
}[script_args.prompt]

squad_dataset = load_dataset("cuad")

train_dataset = squad_dataset["train"].map(instruction)
test_dataset = squad_dataset["test"].map(
    instruction, fn_kwargs={"all_answers": True}
)

train_dataset = filter_long_text_entries(train_dataset)
test_dataset = filter_long_text_entries(test_dataset)

train_dataset = Dataset.from_dict(train_dataset)
test_dataset = Dataset.from_dict(test_dataset)

train_dataset = select_random_subset(train_dataset, 1000)
test_dataset = select_random_subset(test_dataset, 1000)

print(train_dataset[0]["text"])
print(test_dataset[0]["text"])

dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
dataset.save_to_disk(script_args.dataset)
