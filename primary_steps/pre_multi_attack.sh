set -ex

base_dir=$1
last_epoch=$2
target_backbone=$3
attacker_ckpt=${4:-}
target_ckpt=${5:-}


# fix checkpoint to search
checkpoint=${base_dir}/checkpoints/stage1_redLLM/epoch-0/huggingface_model_llama/
# prefer attacker checkpoint for product stage if provided
prod_ckpt=${attacker_ckpt:-$checkpoint}

if [ "$#" -lt 3 ]; then
echo "sh pre_multi_attack.sh base_dir last_epoch target_backbone [attacker_checkpoint] [target_checkpoint]"
exit;
fi

cd ${base_dir}
pip install nltk


############## produce attack data##############
echo "Pre-attack produce attack data"
for i in 0 1 2 3 4 5 6 7
do
{
export CUDA_VISIBLE_DEVICES=${i}
python3 -u scripts/product_attack_data.py \
${prod_ckpt} \
${base_dir}/record/epoch-0/prompt_database \
${base_dir}/record/epoch-0/history_prompt \
${base_dir}/record/epoch-${last_epoch}/prompt_database.${i}.rewrite \
10000

}&
done
wait

cat ${base_dir}/record/epoch-${last_epoch}/prompt_database.*.rewrite | sort -u | shuf > ${base_dir}/record/epoch-${last_epoch}/prompt_database.rewrite

line_count=$(wc -l < "${base_dir}/record/epoch-${last_epoch}/prompt_database.rewrite")

split_line_count=$(expr ${line_count} / 8 + 1)

split -${split_line_count}  ${base_dir}/record/epoch-${last_epoch}/prompt_database.rewrite -d -a 1 ${base_dir}/record/epoch-${last_epoch}/prompt_database.rewrite.rm_duplicate.

for i in 0 1 2 3 4 5 6 7
do
    mv ${base_dir}/record/epoch-${last_epoch}/prompt_database.rewrite.rm_duplicate.${i} ${base_dir}/record/epoch-${last_epoch}/prompt_database.${i}.rewrite.rm_duplicate
done


## romove safety query
for i in 0 1 2 3 4 5 6 7
do
{
export CUDA_VISIBLE_DEVICES=${i}
python3 scripts/infer_safety_pre-attack_llama3.py \
record/epoch-${last_epoch}/prompt_database.${i}.rewrite.rm_duplicate \
record/epoch-${last_epoch}/prompt_database.${i}.rewrite.rm_duplicate.rm_safety
}&
done
wait

#############stage2: red LLM rewrite  ###############
echo "Pre-attack rewrite"
for i in 0 1 2 3 4 5 6 7
do
{
export CUDA_VISIBLE_DEVICES=${i}
python3 scripts/infer_vllm.py \
--checkpoint ${base_dir}/checkpoints/redLLM/epoch-${last_epoch}/huggingface_model_llama/ \
--input ${base_dir}/record/epoch-${last_epoch}/prompt_database.${i}.rewrite.rm_duplicate.rm_safety \
--output ${base_dir}/record/epoch-${last_epoch}/prompt_database.${i}.rewrite.rm_duplicate.rm_safety.attack_output \
--max_tokens 200 \
--model_role red \
--infer_freq 1 \
--temperature 1.0 \
--top_p 0.9 \
--backbone llama3 ${attacker_ckpt:+--checkpoint ${attacker_ckpt}}
}&
done
wait

############ stage3: attack targetLLM and get response ###########
echo "Pre-attack: attack targetLLM and get response"
for i in 0 1 2 3 4 5 6 7
do
{
export CUDA_VISIBLE_DEVICES=${i}
python3 scripts/infer_vllm.py \
--input ${base_dir}/record/epoch-${last_epoch}/prompt_database.${i}.rewrite.rm_duplicate.rm_safety.attack_output \
--output ${base_dir}/record/epoch-${last_epoch}/prompt_database.${i}.rewrite.rm_duplicate.rm_safety.attack_output_response \
--max_tokens 300 \
--model_role target_stage1 \
--infer_freq 1 \
--temperature 0.7 \
--top_p 0.9 \
--backbone ${target_backbone} ${target_ckpt:+--target_checkpoint ${target_ckpt}}
}&
done
wait


########## get guard score ##########
pip3 install transformers==4.43.1
for i in 0 1 2 3 4 5 6 7
do
{
export CUDA_VISIBLE_DEVICES=${i}
python3 -u scripts/infer_llama3.1-guard.py \
        ${base_dir}/record/epoch-${last_epoch}/prompt_database.${i}.rewrite.rm_duplicate.rm_safety.attack_output_response \
        ${base_dir}/record/epoch-${last_epoch}/prompt_database.${i}.rewrite.rm_duplicate.rm_safety.attack_output_response.guard \
        pre_attack_target_stage1
}&
done
wait
pip3 install transformers==4.35.0


######### get helpful score ###########
cd ${base_dir}/DeepSpeedExamples-master/applications/DeepSpeed-Chat/
for i in 0 1 2 3 4 5 6 7
do
{
export CUDA_VISIBLE_DEVICES=${i}

python3 -u infer_ultrarm_gpu.py \
${base_dir}/record/epoch-${last_epoch}/prompt_database.${i}.rewrite.rm_duplicate.rm_safety.attack_output_response.guard \
${base_dir}/record/epoch-${last_epoch}/prompt_database.${i}.rewrite.rm_duplicate.rm_safety.attack_output_response.guard.helpful \
target_pre-attack

}&
done
wait
cd -

cat ${base_dir}/record/epoch-${last_epoch}/prompt_database.*.rewrite.rm_duplicate.rm_safety.attack_output_response.guard.helpful > ${base_dir}/record/epoch-${last_epoch}/prompt_database.rewrite.rm_duplicate.rm_safety.attack_output_response.guard.helpful

python3 scripts/parse_pre-attack_data_v2.py ${base_dir}/record/epoch-${last_epoch}/prompt_database.rewrite.rm_duplicate.rm_safety.attack_output_response.guard.helpful ${base_dir}/record/epoch-${last_epoch}/prompt_database.rewrite.rm_duplicate.rm_safety.attack_output_response.guard.helpful.filter 0.2 0.58

shuf ${base_dir}/data/attack/attack_format2.json | head -3200 > ${base_dir}/record/epoch-${last_epoch}/attack_format2.json.rand3200

cat ${base_dir}/record/epoch-${last_epoch}/attack_format2.json.rand3200 ${base_dir}/record/epoch-${last_epoch}/prompt_database.rewrite.rm_duplicate.rm_safety.attack_output_response.guard.helpful.filter | sort -u | shuf \
> ${base_dir}/record/epoch-${last_epoch}/attack_format2.json.rand3200.cat_pre-attack

line_count=$(wc -l < "${base_dir}/record/epoch-${last_epoch}/attack_format2.json.rand3200.cat_pre-attack")

split_line_count=$(expr ${line_count} / 8 + 1)

split -${split_line_count} ${base_dir}/record/epoch-${last_epoch}/attack_format2.json.rand3200.cat_pre-attack -d -a 1 ${base_dir}/record/epoch-${last_epoch}/attack_format2.json.rand3200.cat_pre-attack.

for i in 0 1 2 3 4 5 6 7
do
cp ${base_dir}/record/epoch-${last_epoch}/attack_format2.json.rand3200.cat_pre-attack.${i} record/epoch-${last_epoch}/attack.json.${i}
done

echo "finish pre-attack!!!"
