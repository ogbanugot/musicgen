import argparse
import gc

import torch
from torch.utils.data import Dataset
import pandas as pd
from audiolm_pytorch import SoundStream, SoundStreamTrainer, SemanticTransformer, SemanticTransformerTrainer
from audiolm_pytorch import HubertWithKmeans, CoarseTransformer, CoarseTransformerTrainer
from audiolm_pytorch import FineTransformer, FineTransformerTrainer
from audiolm_pytorch import EncodecWrapper
import soundfile
from audiolm_pytorch import AudioLM
import torchaudio

the_path = "/home/pythonuser/project/musicgen/songs/vocals_chunk"
training_steps = 10

wav2vec = HubertWithKmeans(
    checkpoint_path='hubert_base_ls960.pt',
    kmeans_path='hubert_base_ls960_L9_km500.bin'
)

encodec = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

semantic_transformer = SemanticTransformer(
    num_semantic_tokens=wav2vec.codebook_size,
    dim=1024,
    depth=6,
    # has_condition=True,  # this will have to be set to True
    # cond_as_self_attn_prefix=True
    # whether to condition as prefix to self attention, instead of cross attention, as was done in 'VALL-E' paper
)

coarse_transformer = CoarseTransformer(
    num_semantic_tokens=wav2vec.codebook_size,
    codebook_size=1024,
    num_coarse_quantizers=3,
    dim=512,
    depth=6,
    flash_attn=False,
)
fine_transformer = FineTransformer(
    num_coarse_quantizers=3,
    num_fine_quantizers=5,
    codebook_size=1024,
    dim=512,
    depth=6,
    flash_attn=False,
)


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

        audio = audio.reshape(-1)

        return caption, audio


def train_encodec():
    trainer = SoundStreamTrainer(
        encodec,
        folder = the_path,
        batch_size = 4,
        grad_accum_every = 8,
        valid_frac=0.1,
        data_max_length = 320 * 32,
        save_results_every = 2,
        save_model_every = int(training_steps/10),
        num_train_steps = training_steps,
        results_folder='./results_encodec',
        force_clear_prev_results=True,
    ).cuda()
    trainer.train()

def train_semantic():
    trainer = SemanticTransformerTrainer(
        transformer=semantic_transformer,
        wav2vec=wav2vec,
        folder=the_path,
        valid_frac=0.1,
        batch_size=4,
        grad_accum_every=8,
        data_max_length_seconds=30,
        save_model_every=int(training_steps/10),
        num_train_steps=training_steps,
        results_folder='./results',
        force_clear_prev_results=True,
    )
    trainer.train()


def train_coarse():
    encodec.load("results_encodec/soundstream.9.pt")
    trainer = CoarseTransformerTrainer(
        transformer=coarse_transformer,
        codec=encodec,
        wav2vec=wav2vec,
        folder=the_path,
        batch_size=2,
        valid_frac=0.1,
        data_max_length_seconds=30,
        save_model_every=int(training_steps/10),
        num_train_steps=training_steps,
        results_folder='./results_coarse',
        force_clear_prev_results=True,
    )
    trainer.train()


def train_fine():
    encodec.load("results_encodec/soundstream.9.pt")
    trainer = FineTransformerTrainer(
        transformer=fine_transformer,
        codec=encodec,
        folder=the_path,
        batch_size=1,
        data_max_length_seconds=20,
        valid_frac=0.1,
        save_model_every=int(training_steps/10),
        num_train_steps=training_steps,
        results_folder='./results_fine',
        force_clear_prev_results=True,
    )

    trainer.train()


def do_inference():
    encodec.load("results_encodec/soundstream.9.pt")
    semantic_transformer.load("results/semantic.transformer.9.pt")
    coarse_transformer.load("results_coarse/coarse.transformer.9.pt")
    fine_transformer.load("results_fine/fine.transformer.9.pt")

    audiolm = AudioLM(
        wav2vec=wav2vec,
        codec=encodec,
        semantic_transformer=semantic_transformer,
        coarse_transformer=coarse_transformer,
        fine_transformer=fine_transformer
    ).cuda()

    generated_wav = audiolm(batch_size=1)
    # wav = generated_wav[0].detach().cpu().numpy()
    # soundfile.write("audiolm_output.wav", wav, samplerate=44100)
    torchaudio.save("audiolm_output.wav", generated_wav.cpu(), 44100)



if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--type", default="semantic")
    # parser.add_argument("--steps", default=training_steps)
    # args = parser.parse_args()

    # if args.type == "semantic":
    #     train_semantic()
    # elif args.type == "coarse":
    #     train_coarse()
    # elif args.type == "fine":
    #     train_fine()
    # elif args.type == "encodec":
    #     train_encodec()
    # elif args.type == "infer":
    #     do_inference()
    train_encodec()
    train_semantic()
    train_coarse()
    train_fine()
    do_inference()
