#!/bin/bash

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org
export https_proxy=http://bj-rd-proxy.byted.org:3128; export http_proxy=http://bj-rd-proxy.byted.org:3128

PROJECT_NAME="on-policy-distillation"
EXPERIMENT_NAME="opd_context_math_qwen3-8b_qwen3-1.7b"
mkdir -p ./${PROJECT_NAME}/${EXPERIMENT_NAME}

data_dir=/opt/tiger/verl/dataset
gsm8k_train_path=$data_dir/gsm8k/train.parquet
gsm8k_test_path=$data_dir/gsm8k/test.parquet
math_train_path=$data_dir/math/train.parquet
math_test_path=$data_dir/math/test.parquet
aime24_path=$data_dir/aime2024/train.parquet
amc23_path=$data_dir/amc2023/test.parquet
train_files="['$math_train_path']"
test_files="['$math_test_path']"
# test_files="['$aime24_path']"
# train_files="['$gsm8k_train_path']"
# test_files="['$gsm8k_test_path']"

export WANDB_API_KEY="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"
export HF_HOME=$(pwd)/hf_cache
export TRANSFORMERS_CACHE=$(pwd)/hf_cache/transformers
export HUGGINGFACE_HUB_CACHE=$(pwd)/hf_cache/hub

unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="max_split_size_mb:256"

python3 -m verl.trainer.main_ppo \
    --config-name opd_trainer \
    actor_rollout_ref2.model.path=Qwen/Qwen3-8B \
    +algorithm.algorithm.teacher_prompt_template_name=context \
    algorithm.adv_estimator=vanilla \
    algorithm.opd.reward_coef=0.0 \
    algorithm.opd.kl_coef=1.0 \
    data.train_files=${train_files} \
    data.val_files=${test_files} \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console", "wandb"]' \
    trainer.critic_warmup=0 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=$(pwd)/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.total_epochs=10 2>&1 | tee ./${PROJECT_NAME}/${EXPERIMENT_NAME}/verl.log \