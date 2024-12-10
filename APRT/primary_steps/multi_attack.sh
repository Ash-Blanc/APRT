set -ex

base_dir=$1
epoch=$2
max_tokens=$3
infer_freq=$4
temperature=$5
top_p=$6
backbone=$7

if [ "$#" -ne 7 ]; then
echo "sh multi_attack.sh base_dir epoch max_tokens infer_freq temperature top_p backbone"
exit;
fi

for i in 0 1 2 3 4 5 6 7
do
{
idx=${i}
export CUDA_VISIBLE_DEVICES=${idx}
python3 scripts/infer_vllm.py \
--checkpoint ${base_dir}/checkpoints/redLLM/epoch-${epoch}/huggingface_model_llama/ \
--input ${base_dir}/record/epoch-${epoch}/attack.json.${idx} \
--output ${base_dir}/record/epoch-${epoch}/attack.json.${idx}.attack_output \
--max_tokens ${max_tokens} \
--model_role red \
--infer_freq ${infer_freq} \
--temperature ${temperature} \
--top_p ${top_p} \
--backbone llama3
}&
done
wait
