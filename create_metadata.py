from essentia_meta import *

from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
import numpy as np

dataset_path = '/home/pythonuser/project/songs'

def filter_predictions(predictions, class_list, threshold=0.1):
    predictions_mean = np.mean(predictions, axis=0)
    sorted_indices = np.argsort(predictions_mean)[::-1]
    filtered_indices = [i for i in sorted_indices if predictions_mean[i] > threshold]
    filtered_labels = [class_list[i] for i in filtered_indices]
    filtered_values = [predictions_mean[i] for i in filtered_indices]
    return filtered_labels, filtered_values

def make_comma_separated_unique(tags):
    seen_tags = set()
    result = []
    result.append("Afrobeats")
    for tag in ', '.join(tags).split(', '):
        if tag not in seen_tags:
            result.append(tag)
            seen_tags.add(tag)
    return ', '.join(result)

def get_audio_features(audio_filename):
    audio = MonoLoader(filename=audio_filename, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    embeddings = embedding_model(audio)

    result_dict = {}

    # predict genres
    genre_model = TensorflowPredict2D(graphFilename="genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
    predictions = genre_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions, genre_labels)
    filtered_labels = ', '.join(filtered_labels).replace("---", ", ").split(', ')
    result_dict['genres'] = make_comma_separated_unique(filtered_labels)

    # predict mood/theme
    mood_model = TensorflowPredict2D(graphFilename="mtg_jamendo_moodtheme-discogs-effnet-1.pb")
    predictions = mood_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions, mood_theme_classes, threshold=0.05)
    result_dict['moods'] = make_comma_separated_unique(filtered_labels)

    # predict instruments
    instrument_model = TensorflowPredict2D(graphFilename="mtg_jamendo_instrument-discogs-effnet-1.pb")
    predictions = instrument_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions, instrument_classes)
    result_dict['instruments'] = filtered_labels

    return result_dict

# Create .jsonl from the extracted features, make a train/test split, and save in the right place.

# set the following variable to True if you want to see a progress bar instead of the printed results:
use_tqdm = False

import os
import json
import random
import librosa
from pydub import AudioSegment
import wave
import re

from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)

# make sure the .jsonl has a place to go
os.makedirs("./egs/train", exist_ok=True)
os.makedirs("./egs/eval", exist_ok=True)

train_len = 0
eval_len = 0

with open("./egs/train/data.jsonl", "w") as train_file, \
     open("./egs/eval/data.jsonl", "w") as eval_file:

    dset = tqdm(os.listdir(dataset_path)) if use_tqdm else os.listdir(dataset_path)
    random.shuffle(dset)

    for filename in dset:
      if filename.endswith(('.mp3', '.wav', '.flac')):
        result = get_audio_features(os.path.join(dataset_path, filename))
        print(f"{filename}: {result}")

        # get key and BPM
        y, sr = librosa.load(os.path.join(dataset_path, filename))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo.round()[0] # not usually accurate lol
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key = np.argmax(np.sum(chroma, axis=1))
        key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key]
        length = librosa.get_duration(y=y, sr=sr)
        # print(f"{filename}: {result}, detected key {key}, detected bpm {tempo}")

        def extract_artist_from_filename(filename):
            match = re.search(r'(.+?)\s\d+_chunk\d+\.wav', filename)
            artist = match.group(1) if match else ""
            return artist.replace("mix", "").strip() if "mix" in artist else artist
        artist_name = extract_artist_from_filename(filename)

        # populate json
        entry = {
            "key": f"{key}",
            "artist": artist_name,
            "sample_rate": 44100,
            "file_extension": "wav",
            "description": "",
            "keywords": "",
            "duration": length,
            "bpm": tempo,
            "genre": result.get('genres', ""),
            "title": "",
            "name": "",
            "instrument": result.get('instruments', ""),
            "moods": result.get('moods', []),
            "path": os.path.join(dataset_path, filename),
        }
        print(entry)

        # train/test split
        if random.random() < 0.85:
            train_len += 1
            train_file.write(json.dumps(entry) + '\n')
        else:
            eval_len += 1
            eval_file.write(json.dumps(entry) + '\n')

print(train_len)
print(eval_len)