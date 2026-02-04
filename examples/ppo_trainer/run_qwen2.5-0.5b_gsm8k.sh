#!/bin/bash

PROJECT_NAME="verl_examples"
EXPERIMENT_NAME="len-.01_ppo_gsm8k_qwen2.5-0.5b"
mkdir -p ./${PROJECT_NAME}/${EXPERIMENT_NAME}

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=./data/gsm8k/train.parquet \
 data.val_files=./data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.rollout.n=1 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=64 \
 custom_reward_function.path=./verl/utils/reward_score/length_penalty.py \
 custom_reward_function.name=compute_length_penalty_reward \
 +custom_reward_function.reward_kwargs.length_penalty_factor=0.01 \
 trainer.project_name=${PROJECT_NAME} \
 trainer.experiment_name=${EXPERIMENT_NAME} \
 trainer.default_local_dir=/data/ziheng_cheng/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME} \
 trainer.logger='["console","wandb"]' \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=1000 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee ./${PROJECT_NAME}/${EXPERIMENT_NAME}/verl.log