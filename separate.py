import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from audio_separator.separator import Separator

dataset_path = "/home/pythonuser/project/dataset"
vocals_path = "/home/pythonuser/project/vocals"


def process_file(filename, separator, vocals_path):
    if filename.endswith(('.mp3', '.wav', '.flac')):
        filename = os.path.join(dataset_path, filename)

        # Perform the separation on specific audio files without reloading the model
        output_files = separator.separate(filename)

        print(f"Separation complete! Output file(s): {' '.join(output_files)}")
        shutil.move(output_files[1], os.path.join(vocals_path, os.path.basename(output_files[1])))


def separate_vocals():
    os.makedirs(vocals_path, exist_ok=True)

    # Initialize the Separator class (with optional configuration properties, below)
    separator = Separator()

    # Load a machine learning model (if unspecified, defaults to 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
    separator.load_model()

    # Get the list of files to process
    files = [f for f in os.listdir(dataset_path) if f.endswith(('.mp3', '.wav', '.flac'))]

    # Use ThreadPoolExecutor to parallelize the file processing
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda f: process_file(f, separator, vocals_path), files), total=len(files)))


if __name__ == '__main__':
    separate_vocals()
