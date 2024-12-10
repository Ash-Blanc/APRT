#!/bin/bash
set -x
set -e
# model args

now_epoch=$1
base_dir=$2

model_path=llama3
freeze_layer=False
sdpa=True
checkpointing=False

data_path=record/epoch-${now_epoch}/sft/red/step3:1.0
max_length=1024

#train args
batch_size=2
global_batch_size=16
learning_rate=1e-5
weight_decay=1e-1
ds_zero_stage=1
save_steps=50000
epochs=5

save_name=epoch-${now_epoch}
save_dir=checkpoints/redLLM

script=scripts/train_sft.py

cd ${base_dir}

deepspeed --num_gpus 8 $script --model_path $model_path --use_sdpa $sdpa --gradient_checkpointing $checkpointing --data_path $data_path --max_length $max_length --batch_size $batch_size --save_dir $save_dir  --save_name $save_name --epochs $epochs --global_batch_size $global_batch_size --learning_rate $learning_rate --weight_decay $weight_decay --ds_zero_stage $ds_zero_stage --save_steps $save_steps
