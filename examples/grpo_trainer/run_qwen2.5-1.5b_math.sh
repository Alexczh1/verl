c#!/bin/bash

PROJECT_NAME="msra"
EXPERIMENT_NAME="TEST_H200_rl_gsm8k_rollout-4_qwen2.5-1.5b"
mkdir -p ./${PROJECT_NAME}/${EXPERIMENT_NAME}

data_dir=/data/ziheng_cheng/dataset
gsm8k_train_path=$data_dir/gsm8k/train.parquet
gsm8k_test_path=$data_dir/gsm8k/test.parquet
# math_train_path=$data_dir/math/train.parquet
# math_test_path=$data_dir/math/test.parquet

# train_files="['$gsm8k_train_path', '$math_train_path']"
# test_files="['$gsm8k_test_path', '$math_test_path']"
train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

# custom_reward_function.path=./verl/utils/reward_score/length_penalty.py \
# custom_reward_function.name=compute_length_penalty_reward \
# +custom_reward_function.reward_kwargs.length_penalty_factor=1.0 \
# +custom_reward_function.reward_kwargs.length_tolerance=50 \

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    +actor_rollout_ref.actor_type="es" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console", "wandb"]' \
    trainer.critic_warmup=0 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=/data/ziheng_cheng/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 2>&1 | tee ./${PROJECT_NAME}/${EXPERIMENT_NAME}/verl.log \