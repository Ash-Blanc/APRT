set -ex

base_dir=$1
now_epoch=$2
max_tokens=$3
infer_freq=$4
temperature=$5
top_p=$6
backbone=$7
target_ckpt=${8:-}
provider=${9:-local}
api_provider=${10:-}
api_model_id=${11:-}

if [ "$#" -lt 7 ]; then
    echo "sh multi_attack.sh base_dir epoch max_tokens infer_freq temperature top_p backbone [target_checkpoint] [provider local|api] [api_provider hf|gemini] [api_model_id]"
    exit;
fi

# Provider args
prov_args="--provider ${provider}"
if [ "${provider}" = "api" ]; then
  [ -n "${api_provider}" ] && prov_args="${prov_args} --api_provider ${api_provider}"
  [ -n "${api_model_id}" ] && prov_args="${prov_args} --api_model_id ${api_model_id}"
fi

for i in 0 1 2 3 4 5 6 7
do
{
idx=${i}
export CUDA_VISIBLE_DEVICES=${idx}
python3 scripts/infer_vllm.py \
--input ${base_dir}/record/epoch-${now_epoch}/advbench_input.json.${idx}.attack_output \
--output ${base_dir}/record/epoch-${now_epoch}/advbench_input.json.${idx}.attack_output_chat_response \
--max_tokens ${max_tokens} \
--model_role target_stage1 \
--infer_freq ${infer_freq} \
--temperature ${temperature} \
--top_p ${top_p} \
--backbone ${backbone} ${target_ckpt:+--target_checkpoint ${target_ckpt}} ${prov_args}
}&
done
wait
