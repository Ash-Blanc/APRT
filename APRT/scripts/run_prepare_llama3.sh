#!/bin/bash
set -x

path=${1}
tokenizer_dir=llama3
save_dir=${2}
num_shards=32
num_proc=32
num_shards=16
num_proc=16
cache_dir=`pwd`/hf_cache
mask_label=True

python3 scripts/prepare_sft.py --data_path $path --tokenizer_dir $tokenizer_dir --save_dir $save_dir/$data --num_shards $num_shards --num_proc $num_proc --cache_dir $cache_dir --mask_label $mask_label --if_llama3 1
