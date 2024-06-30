from audiolm_pytorch import SemanticTransformerTrainer
import soundfile as sf

from audiolm import semantic_transformer, wav2vec, TextAudioDataset


def load_model_for_inference(model, checkpoint_path='results_semantic/semantic.transformer.9.pt'):
    model.load(checkpoint_path)
    model.eval()  # Set the model to evaluation mode
    return model


trainer = SemanticTransformerTrainer(
    transformer=semantic_transformer,
    wav2vec=wav2vec,
    dataset=TextAudioDataset(),
    valid_frac=0.1,
    batch_size=4,
    grad_accum_every=8,
    data_max_length_seconds=30,
    results_folder='./results',
)

model = load_model_for_inference(trainer)

sample = model.generate(text=['afrobeats dance song'], batch_size=1,
                        max_length=128)  # (1, < 128) - may terminate early if it detects [eos]

audio_values = sample[0].detach().cpu().float().numpy()
sf.write("audiolm_out_0.wav", audio_values, 44100)
