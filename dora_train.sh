#!/bin/bash

export USER=og

dora -P audiocraft run \
    "solver=musicgen/musicgen_base_32khz" \
    "model/lm/model_scale=small" \
    "continue_from=//pretrained/facebook/musicgen-small" \
    "conditioner=text2music" \
    "dset=audio/train" \
    "dataset.num_workers=2" \
    "dataset.valid.num_samples=1" \
    "dataset.batch_size=2" \ # CHANGE THIS
    "schedule.cosine.warmup=8" \
    "optim.optimizer=adamw" \ # uses dadaw by default, which is worse for single-gpu runs
    "optim.lr=1e-4" \
    "optim.epochs=5" \ # stops training after 5 epochs- change this
    "optim.updates_per_epoch=1000" \ # 2000 by default, change this if you want checkpoints quicker ig
    "optim.adam.weight_decay=0.01" \
    "generate.lm.prompted_samples=False" \ # skip super long generate step
    "generate.lm.gen_gt_samples=True"
