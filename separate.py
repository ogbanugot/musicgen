import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from audio_separator.separator import Separator

dataset_path = "/home/pythonuser/project/dataset"
vocals_path = "/home/pythonuser/project/vocals"


def process_file(filename, separator, vocals_path, log_path, filenames_list):
    try:
        if filename.endswith(('.mp3', '.wav', '.flac')):
            filepath = os.path.join(dataset_path, filename)
            print(filenames_list[0:10])
            print("filename########", filename)
            if filename in filenames_list:
                print(f"skipping: {filename}")
                return
            # Perform the separation on specific audio files without reloading the model
            output_files = separator.separate(filepath)

            print(f"Separation complete! Output file(s): {' '.join(output_files)}")
            shutil.move(output_files[1], os.path.join(vocals_path, os.path.basename(output_files[1])))
    except Exception as e:
        print(f"Failed to process file: {filename}. Error: {str(e)}")
        with open(log_path, 'a') as log_file:
            log_file.write(f"{filename}\n")


def separate_vocals():
    os.makedirs(vocals_path, exist_ok=True)

    # Initialize the Separator class (with optional configuration properties, below)
    separator = Separator()

    # Load a machine learning model (if unspecified, defaults to 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
    separator.load_model()
    log_path = os.path.join(vocals_path, 'failed_files.txt')
    slog_file = "separated.txt"

    # Initialize an empty list to store the filenames
    filenames_list = []

    # Open the log file in read mode
    with open(slog_file, "r") as log:
        # Iterate over each line in the file
        for line in log:
            # Strip any leading/trailing whitespace (like newline characters) and add to the list
            filenames_list.append(line.strip())

    # Get the list of files to process
    files = [f for f in os.listdir(dataset_path) if f.endswith(('.mp3', '.wav', '.flac'))]

    # Use ThreadPoolExecutor to parallelize the file processing
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda f: process_file(f, separator, vocals_path, log_path, filenames_list), files), total=len(files)))


if __name__ == '__main__':
    separate_vocals()
