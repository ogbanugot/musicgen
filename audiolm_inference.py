import torch
from audiolm import semantic_transformer


def load_model_for_inference(model, checkpoint_path='model_checkpoint.pth'):
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # Set the model to evaluation mode
    return model


model_for_inference = semantic_transformer
model = load_model_for_inference(model_for_inference, 'model_checkpoint.pth')

sample = model.generate(text=['sound of rain drops on the rooftops'], batch_size=1,
                        max_length=2)  # (1, < 128) - may terminate early if it detects [eos]
