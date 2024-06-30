import torch
from audiolm import trainer


def load_model_for_inference(model, checkpoint_path='model_checkpoint.pth'):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model


model_for_inference = trainer
model = load_model_for_inference(model_for_inference, 'audiolm_checkpoint.pth')

sample = model.generate(text=['sound of rain drops on the rooftops'], batch_size=1,
                        max_length=2)  # (1, < 128) - may terminate early if it detects [eos]
