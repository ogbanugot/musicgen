#!/bin/bash

export USER=og

export USER=og; dora -P audiocraft run "solver=musicgen/musicgen_base_32khz" "model/lm/model_scale=small" "continue_from=//pretrained/facebook/musicgen-small" "conditioner=text2music" "dset=audio/train" "dataset.num_workers=2" "dataset.valid.num_samples=1" "dataset.batch_size=2" "schedule.cosine.warmup=8" "optim.optimizer=adamw" "optim.lr=1e-4" "optim.epochs=5" "optim.updates_per_epoch=1000" "optim.adam.weight_decay=0.01" "generate.lm.prompted_samples=False" "generate.lm.gen_gt_samples=True"

