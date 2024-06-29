import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from audiolm_pytorch import HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer

# https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
# https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
# https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin

wav2vec = HubertWithKmeans(
    checkpoint_path='./hubert/hubert_base_ls960.pt',
    kmeans_path='./hubert/hubert_base_ls960_L9_km500.bin',
    target_sample_hz=41000,
)

semantic_transformer = SemanticTransformer(
    num_semantic_tokens=500,
    dim=1024,
    depth=6,
    has_condition=True,  # this will have to be set to True
    cond_as_self_attn_prefix=True
    # whether to condition as prefix to self attention, instead of cross attention, as was done in 'VALL-E' paper
).cuda()


class TextAudioDataset(Dataset):
    def __init__(self, csv_file='audiolm_dataset.csv', sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_length = self.sample_rate * 30  # 30 seconds of audio
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx]['audio']
        caption = self.data.iloc[idx]['caption']

        # Load audio file
        audio, sr = torchaudio.load(audio_path)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio = resampler(audio)

        # Ensure audio length matches self.audio_length
        if audio.size(1) > self.audio_length:
            audio = audio[:, :self.audio_length]
        elif audio.size(1) < self.audio_length:
            padding = self.audio_length - audio.size(1)
            audio = torch.nn.functional.pad(audio, (0, padding))

        return caption, audio


def save_checkpoint(model, optimizer, filename='audiolm_checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename='audiolm_checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Function to train with checkpoint saving
def train_with_checkpoint(trainer, save_interval, checkpoint_path='checkpoint.pth'):
    for step in range(trainer.num_train_steps):
        trainer.train_step()

        # Save checkpoint at the defined interval
        if step % save_interval == 0 and step > 0:
            save_checkpoint(trainer.transformer, trainer.optimizer, checkpoint_path)
            print(f"Checkpoint saved at step {step}")


dataset = TextAudioDataset()

# instantiate semantic transformer trainer and train
trainer = SemanticTransformerTrainer(
    transformer=semantic_transformer,
    wav2vec=wav2vec,
    dataset=dataset,
    batch_size=4,
    grad_accum_every=8,
    data_max_length_seconds=30,
    num_train_steps=1_000_000
)

# Define the save interval (e.g., save every 1000 steps)
save_interval = 1000

checkpoint_path = 'audiolm_checkpoint.pth'

train_with_checkpoint(trainer, save_interval, checkpoint_path)
