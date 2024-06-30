import csv
import os
from tqdm import tqdm

dataset_path = '/home/pythonuser/project/musicgen/songs/vocals_chunks'
# dataset_path = 'songs/vocals_chunks'

dset = tqdm(os.listdir(dataset_path))
captions = {
    1: "afrobeats, slow, chill, relaxing, deep, male, man, guy",
    2: "afrobeats, happy, love, funny, female, woman, babe, colab",
    3: "afrobeats, dance, party, man, male, shaye, flex",
    4: "afrobeats, love, dance, guy, male, man, rhythm",
    5: "afrobeats, guy, love, chill, good lyrics",
    6: "afrobeats, guy, yoruba, grind, nice beat, chill, pidgin",
    7: "afrobeats, woman, babe, sex, love, good beats",
    8: "afrobeats, amapiano, dance, nice beat, party, guy, male",
    9: "afrobeats, banging beat, dance, party, guy, male",
    10: "afrobeats, woman, babe, grind, chill, flex, rhythm, relax",

}
headers = ['audio', 'caption']
with open('audiolm_dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for filename in dset:
        if filename.endswith(('.mp3', '.wav', '.flac')):
            index = int(str(filename).split("_")[0])
            caption = captions[index]
            path = os.path.join(dataset_path, filename)
            result = [path, caption]
            writer.writerow(result)
