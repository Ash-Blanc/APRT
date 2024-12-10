set -ex


epoch=$1
base_dir=$2
model_role=$3

if [ "$#" -ne 3 ]; then
   echo "sh compute_helpfulness.sh epoch base_dir model_role"
fi

cd ${base_dir}/DeepSpeedExamples-master/applications/DeepSpeed-Chat/
for i in 0 1 2 3 4 5 6 7
do
{
export CUDA_VISIBLE_DEVICES=${i}
python3 -u infer_ultrarm_gpu.py \
${base_dir}/record/epoch-${epoch}/advbench_input.json.${i}.attack_output_chat_response.guard \
${base_dir}/record/epoch-${epoch}/advbench_input.json.${i}.attack_output_chat_response.guard.helpful \
${model_role}

}&
done
wait

cd -
