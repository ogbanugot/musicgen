import os
import json
import pandas as pd
from tqdm import tqdm

# Load the CSV file
csv_path = 'audiolm_dataset.csv'
data = pd.read_csv(csv_path)

# Define paths
root_dir = './AudioLDM-training-finetuning/data'
audioset_dir = os.path.join(root_dir, 'dataset/audioset')
metadata_dir = os.path.join(root_dir, 'dataset/metadata/audiocaps')
datafiles_dir = os.path.join(metadata_dir, 'datafiles')
testset_subset_dir = os.path.join(metadata_dir, 'testset_subset')

# Create directories if they don't exist
os.makedirs(audioset_dir, exist_ok=True)
os.makedirs(datafiles_dir, exist_ok=True)
os.makedirs(testset_subset_dir, exist_ok=True)

# Copy audio files to the audioset directory
for audio_file in tqdm(data['audio']):
    file_name = os.path.basename(audio_file)
    new_path = os.path.join(audioset_dir, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    os.system(f'cp {audio_file} {new_path}')

# Create metadata JSON files
train_data = []
test_data = []

for i, row in data.iterrows():
    datapoint = {
        'wav': os.path.basename(row['audio']),
        'caption': row['caption']
    }
    # You can define your own condition to split between train and test
    if i % 5 == 0:  # Example condition
        test_data.append(datapoint)
    else:
        train_data.append(datapoint)

# Save the train metadata
train_metadata = {'data': train_data}
with open(os.path.join(datafiles_dir, 'audiocaps_train_label.json'), 'w') as f:
    json.dump(train_metadata, f, indent=4)

# Save the test metadata
test_metadata = {'data': test_data}
with open(os.path.join(testset_subset_dir, 'audiocaps_test_nonrepeat_subset_0.json'), 'w') as f:
    json.dump(test_metadata, f, indent=4)

# Save the dataset root metadata
dataset_root_metadata = {
    'dataset_root': {
        'audioset': 'dataset/audioset',
        'metadata': 'dataset/metadata'
    }
}
with open(os.path.join(metadata_dir, 'dataset_root.json'), 'w') as f:
    json.dump(dataset_root_metadata, f, indent=4)

print("Dataset structured successfully!")
