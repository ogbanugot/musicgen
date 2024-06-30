import os
import shutil
from audio_separator.separator import Separator

dataset_path = "/home/pythonuser/project/musicgen/songs/original_new"
vocals_path = "/home/pythonuser/project/musicgen/songs/vocals"


def separate_vocals():
    os.makedirs(vocals_path, exist_ok=True)

    # Initialize the Separator class (with optional configuration properties, below)
    separator = Separator()

    # Load a machine learning model (if unspecified, defaults to 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
    separator.load_model()

    for filename in os.listdir(dataset_path):
        if filename.endswith(('.mp3', '.wav', '.flac')):
            filename = os.path.join(dataset_path, filename)

            # Perform the separation on specific audio files without reloading the model
            output_files = separator.separate(filename)

            print(f"Separation complete! Output file(s): {' '.join(output_files)}")
            shutil.copy(output_files[1], os.path.join(vocals_path, output_files[1]))


if __name__ == '__main__':
    separate_vocals()
