import argparse
from peft import PeftConfig, PeftModel
from transformers import AutoModelForTextToWaveform, AutoProcessor
import torch
import soundfile as sf

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Generate audio from text using a pretrained model.")
parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate.")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")

repo_id = "ogbanugot/musicgen-small-lora-afrobeats"

config = PeftConfig.from_pretrained(repo_id)
model = AutoModelForTextToWaveform.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, repo_id).to(device)

processor = AutoProcessor.from_pretrained(repo_id)

inputs = processor(
    text=["afrobeats love song", "afrobeats"],
    padding=True,
    return_tensors="pt",
).to(device)

audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=args.max_new_tokens)

sampling_rate = model.config.audio_encoder.sampling_rate
audio_values = audio_values.cpu().float().numpy()

sf.write("musicgen_out_0.wav", audio_values[0].T, sampling_rate)
sf.write("musicgen_out_1.wav", audio_values[1].T, sampling_rate)
