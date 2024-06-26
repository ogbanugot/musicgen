# Make the .yaml

config_path = "./audiocraft/config/dset/audio/train.yaml"

# point to the folders that your .jsonl is in
data_path = "egs/train"
eval_data_path = "egs/eval"

package = "package" # yay python not letting me put #@.package in a string :/
yaml_contents = f"""#@{package} __global__

device: cpu

datasource:
  max_channels: 2
  max_sample_rate: 44100

  evaluate: {eval_data_path}
  generate: {data_path}
  train: {data_path}
  valid: {eval_data_path}
"""

with open(config_path, 'w') as yaml_file:
    yaml_file.write(yaml_contents)