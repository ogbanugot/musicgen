import csv
import os
import random
from tqdm import tqdm
from datasets import DatasetDict, Audio

dataset_path = '/home/pythonuser/project/songs'

dset = tqdm(os.listdir(dataset_path))

headers = ['audio']
with open('dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for filename in dset:
      for i in range(6):
        if filename.endswith(('.mp3', '.wav', '.flac')):
            path = os.path.join(dataset_path, filename)
            result = [path]
            writer.writerow(result)

dataset = DatasetDict.from_csv({"train": "dataset.csv"})
dataset = dataset.cast_column("audio", Audio())
# dataset.save_to_disk(".")
dataset.push_to_hub("ogbanugot/musicgen")
