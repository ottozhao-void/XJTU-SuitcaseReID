#!/bin/bash

# Set visible devices to GPUs 4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Launch training with PyTorch distributed mode
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 \
    strong_baseline_suitcase.py --config suitcase_config.yaml --launcher pytorch