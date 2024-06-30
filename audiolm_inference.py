import torch
from audiolm_pytorch import SemanticTransformer, HubertWithKmeans, SemanticTransformerTrainer

wav2vec = HubertWithKmeans(
    checkpoint_path='hubert_base_ls960.pt',
    kmeans_path='hubert_base_ls960_L9_km500.bin',
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

# instantiate semantic transformer trainer and train
trainer = SemanticTransformerTrainer(
    transformer=semantic_transformer,
    wav2vec=wav2vec,
    valid_frac=0.1,
    batch_size=4,
    grad_accum_every=8,
    data_max_length_seconds=30,
    num_train_steps=10
)


def load_model_for_inference(model, checkpoint_path='model_checkpoint.pth'):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model


model_for_inference = semantic_transformer
model = load_model_for_inference(model_for_inference, 'audiolm_checkpoint.pth')

sample = model.generate(text=['sound of rain drops on the rooftops'], batch_size=1,
                        max_length=2)  # (1, < 128) - may terminate early if it detects [eos]
