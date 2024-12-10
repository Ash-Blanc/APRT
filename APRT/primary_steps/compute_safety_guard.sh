set -ex


epoch=$1
target_stage=$2

if [ "$#" -ne 2 ]; then
   echo "sh compute_safety_guard.sh epoch target_stage"
fi

for i in 0 1 2 3 4 5 6 7
do
{
export CUDA_VISIBLE_DEVICES=${i}
python3 -u scripts/infer_llama3.1-guard.py \
	record/epoch-${epoch}/attack.json.${i}.attack_output_response \
	record/epoch-${epoch}/attack.json.${i}.attack_output_response.guard \
	${target_stage}
}&
done
wait
