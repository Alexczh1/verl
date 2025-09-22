PROJECT_NAME="msra-sft"
EXPERIMENT_NAME="eswp_prune_.2_H200_numina_qwen2.5-0.5b"
mkdir -p ./${PROJECT_NAME}/${EXPERIMENT_NAME}

save_path=/home/exouser/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

data_dir=/home/exouser/data
train_path=$data_dir/numina_cot/train.parquet
test_path=$data_dir/numina_cot/test.parquet

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
     -m verl.trainer.fsdp_sft_trainer_eswp \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.max_length=1024 \
    data.truncation=right \
    data.train_batch_size=32 \
    +data.train_mini_batch_size=8 \
    data.micro_batch_size_per_gpu=8 \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B \
    model.fsdp_config.model_dtype=bf16 \
    optim.lr=5e-5 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=10 \
    trainer.n_gpus_per_node=1 \
    trainer.save_freq=20000 \
    trainer.test_freq=200 \
    +trainer.prune_ratio=0.2 \
    +trainer.beta_1=0.2 \
    +trainer.beta_2=0.9 \
    +trainer.stop_threshold=8 \
    +trainer.start_threshold=0 \
    trainer.logger='["console", "wandb"]' 2>&1 | tee ./${PROJECT_NAME}/${EXPERIMENT_NAME}/verl.log \