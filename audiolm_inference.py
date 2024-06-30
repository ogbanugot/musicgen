import torch
from audiolm_pytorch import SemanticTransformerWrapper
import soundfile as sf

from audiolm import semantic_transformer, wav2vec


def load_model_for_inference(model, checkpoint_path='model_checkpoint.pth'):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model


model_for_inference = SemanticTransformerWrapper(
        transformer=semantic_transformer,
        wav2vec=wav2vec,
    )

model = load_model_for_inference(model_for_inference, 'audiolm_checkpoint.pth')

sample = model.generate(text=['sound of rain drops on the rooftops'], batch_size=1,
                        max_length=128)  # (1, < 128) - may terminate early if it detects [eos]

audio_values = sample[0].detach().cpu().numpy()
sf.write("audiolm_out_0.wav", audio_values, 44100)
