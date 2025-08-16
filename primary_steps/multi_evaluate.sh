set -ex

base_dir=$1
now_epoch=$2
max_tokens=$3
infer_freq=$4
temperature=$5
top_p=$6
backbone=$7
attacker_ckpt=${8:-}
provider=${9:-local}
api_provider=${10:-}
api_model_id=${11:-}

if [ "$#" -lt 7 ]; then
echo "sh multi_attack.sh base_dir now_epoch max_tokens infer_freq temperature top_p backbone [attacker_checkpoint] [provider local|api] [api_provider hf|gemini] [api_model_id]"
exit;
fi

# Choose checkpoint argument for attacker
if [ -n "${attacker_ckpt}" ]; then
  ckpt_arg="--checkpoint ${attacker_ckpt}"
else
  ckpt_arg="--checkpoint ${base_dir}/checkpoints/redLLM/epoch-${now_epoch}/huggingface_model_llama/"
fi

# Provider args
env_args="--provider ${provider}"
if [ "${provider}" = "api" ]; then
  [ -n "${api_provider}" ] && env_args="${env_args} --api_provider ${api_provider}"
  [ -n "${api_model_id}" ] && env_args="${env_args} --api_model_id ${api_model_id}"
fi

for i in 0 1 2 3 4 5 6 7
do
{
idx=${i}
export CUDA_VISIBLE_DEVICES=${idx}
python3 scripts/infer_vllm.py \
${ckpt_arg} \
--input ${base_dir}/data/testset/advbench/advbench_input.json.${idx} \
--output ${base_dir}/record/epoch-${now_epoch}/advbench_input.json.${idx}.attack_output \
--max_tokens ${max_tokens} \
--model_role red \
--infer_freq ${infer_freq} \
--temperature ${temperature} \
--top_p ${top_p} \
--backbone ${backbone} ${env_args}
}&
done
wait
