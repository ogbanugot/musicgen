import argparse
from contextlib import nullcontext
from functools import partial

import torch
from audiolm_pytorch.trainer import accum_log, exists, dict_values_to_device, noop
from torch.utils.data import Dataset
import pandas as pd
from audiolm_pytorch import SemanticTransformer, SemanticTransformerTrainer
from audiolm_pytorch import HubertWithKmeans, CoarseTransformer, CoarseTransformerTrainer
from audiolm_pytorch import FineTransformer, FineTransformerTrainer
from audiolm_pytorch import EncodecWrapper
import soundfile
from audiolm_pytorch import AudioLM
import torchaudio

the_path = "/home/pythonuser/project/musicgen/songs/vocals_chunks"
training_steps = 10

wav2vec = HubertWithKmeans(
    checkpoint_path='hubert_base_ls960.pt',
    kmeans_path='hubert_base_ls960_L9_km500.bin'
)

encodec = EncodecWrapper()

semantic_transformer = SemanticTransformer(
    num_semantic_tokens=500,
    dim=1024,
    depth=6,
    has_condition=True,  # this will have to be set to True
    cond_as_trainer_attn_prefix=True
    # whether to condition as prefix to trainer attention, instead of cross attention, as was done in 'VALL-E' paper
)

coarse_transformer = CoarseTransformer(
    num_semantic_tokens=wav2vec.codebook_size,
    codebook_size=1024,
    num_coarse_quantizers=3,
    dim=512,
    depth=6,
    flash_attn=False
)
fine_transformer = FineTransformer(
    num_coarse_quantizers=3,
    num_fine_quantizers=5,
    codebook_size=1024,
    dim=512,
    depth=6,
    flash_attn=True
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


def train_semantic():
    trainer = SemanticTransformerTrainer(
        transformer=semantic_transformer,
        wav2vec=wav2vec,
        dataset=TextAudioDataset(),
        valid_frac=0.1,
        batch_size=4,
        grad_accum_every=8,
        data_max_length_seconds=30,
        num_train_steps=training_steps,
        results_folder='./results',
    )

    def train_step(trainer):
        device = trainer.device

        steps = int(trainer.steps.item())

        trainer.transformer.train()

        # logs
        logs = {}

        # update transformer
        for i in range(trainer.grad_accum_every):
            is_last = i == (trainer.grad_accum_every - 1)
            context = partial(trainer.accelerator.no_sync, trainer.train_wrapper) if not is_last else nullcontext

            data_kwargs = trainer.data_tuple_to_kwargs(next(trainer.dl_iter))

            with trainer.accelerator.autocast(), context():
                loss = trainer.train_wrapper(**data_kwargs, return_loss=True)

                trainer.accelerator.backward(loss / trainer.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / trainer.grad_accum_every})

        if trainer.max_grad_norm is not None:
            trainer.accelerator.clip_grad_norm_(trainer.transformer.parameters(), trainer.max_grad_norm)

        trainer.optim.step()
        trainer.optim.zero_grad()

        # log
        trainer.print(f"{steps}: loss: {logs['loss']}")
        trainer.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # sample results every so often
        trainer.accelerator.wait_for_everyone()

        if trainer.is_main and not (steps % trainer.save_results_every):
            valid_loss = 0
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.train_wrapper)

            for _ in range(trainer.average_valid_loss_over_grad_accum_every):
                data_kwargs = trainer.data_tuple_to_kwargs(next(trainer.valid_dl_iter))
                data_kwargs = dict_values_to_device(data_kwargs, unwrapped_model.device)

                with torch.inference_mode():
                    unwrapped_model.eval()
                    valid_loss += unwrapped_model(**data_kwargs, return_loss=True).clone().detach()

            valid_loss /= trainer.average_valid_loss_over_grad_accum_every

            trainer.print(f'{steps}: valid loss {valid_loss}')
            trainer.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # save model every so often
        if trainer.is_main and not (steps % trainer.save_model_every):
            model_path = str(trainer.results_folder / f'semantic.transformer.{steps}.pt')
            trainer.save(model_path)
            trainer.print(f'{steps}: saving model to {str(trainer.results_folder)}')

        trainer.accelerator.wait_for_everyone()

        trainer.steps.add_(1)
        return logs

    def train(trainer, log_fn=noop):

        while trainer.steps < trainer.num_train_steps:
            print(trainer.steps)
            logs = train_step(trainer)
            log_fn(logs)
        trainer.print('training complete')

    train(trainer)


def train_coarse():
    trainer = CoarseTransformerTrainer(
        transformer=coarse_transformer,
        codec=encodec,
        wav2vec=wav2vec,
        folder=the_path,
        batch_size=64,
        data_max_length_seconds=30,
        num_train_steps=training_steps,
        results_folder='./results_coarse',
    )

    def train_step(trainer):
        device = trainer.device

        steps = int(trainer.steps.item())

        trainer.transformer.train()

        # logs

        logs = {}

        # update transformer

        for i in range(trainer.grad_accum_every):
            is_last = i == (trainer.grad_accum_every - 1)
            context = partial(trainer.accelerator.no_sync, trainer.train_wrapper) if not is_last else nullcontext

            data_kwargs = dict(zip(trainer.ds_fields, next(trainer.dl_iter)))

            with trainer.accelerator.autocast(), context():
                loss = trainer.train_wrapper(
                    **data_kwargs,
                    return_loss=True
                )

                trainer.accelerator.backward(loss / trainer.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / trainer.grad_accum_every})

        if exists(trainer.max_grad_norm):
            trainer.accelerator.clip_grad_norm_(trainer.transformer.parameters(), trainer.max_grad_norm)

        trainer.optim.step()
        trainer.optim.zero_grad()

        # log

        trainer.print(f"{steps}: loss: {logs['loss']}")
        trainer.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # sample results every so often

        trainer.accelerator.wait_for_everyone()

        if trainer.is_main and not (steps % trainer.save_results_every):
            valid_loss = 0
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.train_wrapper)

            for i in range(trainer.average_valid_loss_over_grad_accum_every):
                data_kwargs = dict(zip(trainer.ds_fields, next(trainer.valid_dl_iter)))
                data_kwargs = dict_values_to_device(data_kwargs, unwrapped_model.device)

                with torch.no_grad():
                    unwrapped_model.eval()

                    valid_loss += unwrapped_model(
                        **data_kwargs,
                        return_loss=True
                    )

            valid_loss = valid_loss.clone()  # avoid inference mode to non-inference mode error
            valid_loss /= trainer.average_valid_loss_over_grad_accum_every

            trainer.print(f'{steps}: valid loss {valid_loss}')
            trainer.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # save model every so often

        if trainer.is_main and not (steps % trainer.save_model_every):
            model_path = str(trainer.results_folder / f'coarse.transformer.{steps}.pt')
            trainer.save(model_path)
            trainer.print(f'{steps}: saving model to {str(trainer.results_folder)}')

        trainer.accelerator.wait_for_everyone()

        trainer.steps.add_(1)
        return logs

    def train(trainer, log_fn=noop):

        while trainer.steps < trainer.num_train_steps:
            logs = train_step(trainer)
            log_fn(logs)

        trainer.print('training complete')

    train(trainer)


def train_fine():
    trainer = FineTransformerTrainer(
        transformer=fine_transformer,
        codec=encodec,
        folder=the_path,
        batch_size=64,
        data_max_length=30,
        num_train_steps=training_steps,
        results_folder='./results_fine',
    )

    def train_step(trainer):
        device = trainer.device

        steps = int(trainer.steps.item())

        trainer.transformer.train()

        # logs

        logs = {}

        # update transformer

        for i in range(trainer.grad_accum_every):
            is_last = i == (trainer.grad_accum_every - 1)
            context = partial(trainer.accelerator.no_sync, trainer.train_wrapper) if not is_last else nullcontext

            data_kwargs = trainer.data_tuple_to_kwargs(next(trainer.dl_iter))

            with trainer.accelerator.autocast(), context():
                loss = trainer.train_wrapper(**data_kwargs, return_loss=True)

                trainer.accelerator.backward(loss / trainer.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / trainer.grad_accum_every})

        if exists(trainer.max_grad_norm):
            trainer.accelerator.clip_grad_norm_(trainer.transformer.parameters(), trainer.max_grad_norm)

        trainer.optim.step()
        trainer.optim.zero_grad()

        # log

        trainer.print(f"{steps}: loss: {logs['loss']}")
        trainer.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # sample results every so often

        trainer.accelerator.wait_for_everyone()

        if trainer.is_main and not (steps % trainer.save_results_every):
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.train_wrapper)
            valid_loss = 0

            for i in range(trainer.average_valid_loss_over_grad_accum_every):
                data_kwargs = trainer.data_tuple_to_kwargs(next(trainer.valid_dl_iter))
                data_kwargs = dict_values_to_device(data_kwargs, unwrapped_model.device)

                with torch.inference_mode():
                    unwrapped_model.eval()
                    valid_loss += unwrapped_model(**data_kwargs, return_loss=True)

            valid_loss = valid_loss.clone()  # avoid inference mode to non-inference mode error
            valid_loss /= trainer.average_valid_loss_over_grad_accum_every

            trainer.print(f'{steps}: valid loss {valid_loss}')
            trainer.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # save model every so often

        if trainer.is_main and not (steps % trainer.save_model_every):
            model_path = str(trainer.results_folder / f'fine.transformer.{steps}.pt')
            trainer.save(model_path)
            trainer.print(f'{steps}: saving model to {str(trainer.results_folder)}')

        trainer.accelerator.wait_for_everyone()

        trainer.steps.add_(1)
        return logs

    def train(trainer, log_fn=noop):

        while trainer.steps < trainer.num_train_steps:
            logs = train_step(trainer)
            log_fn(logs)

        trainer.print('training complete')

    train(trainer)


def do_inference():
    semantic_transformer.load("results/semantic.transformer.405000.pt")
    coarse_transformer.load("results_coarse/coarse.transformer.683000.pt")
    fine_transformer.load("results_fine/fine.transformer.828000.pt")

    audiolm = AudioLM(
        wav2vec=wav2vec,
        codec=encodec,
        semantic_transformer=semantic_transformer,
        coarse_transformer=coarse_transformer,
        fine_transformer=fine_transformer
    ).cuda()

    generated_wav = audiolm(batch_size=1)
    wav = generated_wav[0].detach().cpu().numpy()
    soundfile.write("output.wav", wav, samplerate=44100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="semantic")
    args = parser.parse_args()

    if args.type == "semantic":
        train_semantic()
    elif args.type == "coarse":
        train_coarse()
    elif args.type == "fine":
        train_fine()
    elif args.type == "infer":
        do_inference()
