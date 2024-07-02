# split and resample

import os
import shutil
from pydub import AudioSegment

dataset_path = "/home/pythonuser/project/musicgen/songs/vocals"
originals = "/home/pythonuser/project/musicgen/songs/original"


def rename_originals():
    os.makedirs("/home/pythonuser/project/musicgen/songs/original_new", exist_ok=True)
    for filename in os.listdir(originals):
        if filename.endswith(('.mp3', '.wav', '.flac')):
            original = os.path.join(originals, filename)
            new = os.path.join("/home/pythonuser/project/musicgen/songs/original_new", filename.strip("songs "))
            print(new)
            shutil.copy(original, new)


def segment():
    path = "/home/pythonuser/project/musicgen/songs/vocals_chunks"
    os.makedirs(path, exist_ok=True)
    for filename in os.listdir(dataset_path):
        print(filename)
        if filename.endswith(('.mp3', '.wav', '.flac')):

            original = os.path.join(dataset_path, filename)

            audio = AudioSegment.from_file(f"{original}")

            # resample
            audio = audio.set_frame_rate(44100)

            # split into 30-second chunks
            for i in range(0, len(audio), 30000):
                chunk = audio[i:i + 30000]
                export = os.path.join(path, f"{filename[:-4]}_chunk{i // 1000}.wav")
                chunk.export(export, format="wav")


if __name__ == '__main__':
    rename_originals()
