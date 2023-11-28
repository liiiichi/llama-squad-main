from datasets import DatasetDict, load_dataset, load_from_disk
import pandas as pd

ds = load_from_disk("data/squad_v2")["train"]
df = ds.to_pandas()

print(ds['text'][0])

