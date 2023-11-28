# from transformers import DataCollatorForLanguageModeling
# import torch

# class SquadDataCollator(DataCollatorForLanguageModeling):
#     def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
#             super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
#             # Assuming the tokenizer has been updated with the custom tokens
#             self.answer_start_token_id = tokenizer.convert_tokens_to_ids("<answer_start>")
#             self.answer_end_token_id = tokenizer.convert_tokens_to_ids("<answer_end>")

#     def __call__(self, examples):
#         batch = super().__call__(examples)
#         print(self.answer_start_token_id)
#         print(self.answer_end_token_id)
#         debug_file = open("debug_labels.txt", "w")


#         # Only apply cross entropy loss to the answer part of the labels
#         for idx, label in enumerate(batch["labels"]):
#             print(batch["labels"])
#             # print(label)
#             # print(label.tolist())
#             debug_file.write(f"Label {idx}: {label.tolist()}\n")
#             debug_file.close()
#             # Find the positions of the start and end tokens
#             answer_end = torch.where(label == self.answer_end_token_id)[0][0]
#             answer_start = torch.where(label == self.answer_start_token_id)[0][-1]
#             # answer_start_positions = (label == self.answer_start_token_id).nonzero(as_tuple=True)[0]
#             # answer_end_positions = (label == self.answer_end_token_id).nonzero(as_tuple=True)[0]

#             answer_start = answer_start_positions[0]
#             answer_end = answer_end_positions[0]
#             label[:answer_start] = -100
#             label[answer_end + 1:] = -2
#             batch["labels"][idx] = label

#         return batch

from transformers import DataCollatorForLanguageModeling
import torch

class SquadDataCollator(DataCollatorForLanguageModeling):
    answer_start_token_id = 7521  # "_```"

    def __call__(self, examples):
        batch = super().__call__(examples)

        # Only apply cross entropy loss to the answer part of the labels
        for idx, label in enumerate(batch["labels"]):
            answer_end = torch.where(label == -100)[0][0]
            answer_start = torch.where(label == self.answer_start_token_id)[0][-1]
            label[:answer_start] = -100
            label[answer_end] = 2
            batch["labels"][idx] = label

        return batch
