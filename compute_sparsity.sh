#!/bin/bash

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org
export https_proxy=http://bj-rd-proxy.byted.org:3128; export http_proxy=http://bj-rd-proxy.byted.org:3128

export WANDB_API_KEY="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"
export HF_HOME=$(pwd)/hf_cache
export TRANSFORMERS_CACHE=$(pwd)/hf_cache/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/hf_cache/hub

unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="max_split_size_mb:256"

python scripts/compare_checkpoint_layers.py --checkpoint /opt/tiger/verl/checkpoints/on-policy-distillation/gsm8k-math_qwen3-1.7b_grpo_rollout-8/global_step_1160/actor --hf_model Qwen/Qwen3-1.7B --output diff.json  --only-params