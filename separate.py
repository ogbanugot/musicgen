import os
from audio_separator.separator import Separator

dataset_path = "songs"


def separate_vocals():
    # Initialize the Separator class (with optional configuration properties, below)
    separator = Separator()

    # Load a machine learning model (if unspecified, defaults to 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
    separator.load_model()

    filename = os.path.join(dataset_path, "original/song 10.mp3")

    # Perform the separation on specific audio files without reloading the model
    output_files = separator.separate(filename)

    print(f"Separation complete! Output file(s): {' '.join(output_files)}")
