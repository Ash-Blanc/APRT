set -ex

mkdir checkpoints record

cp -r env/DeepSpeedExamples-master .
# Please manually download https://huggingface.co/openbmb/UltraRM-13b 
mdkir -p DeepSpeedExamples-master/applications/DeepSpeed-Chat/openbmb/UltraRM-13b
cp -r UltraRM-13b DeepSpeedExamples-master/applications/DeepSpeed-Chat/openbmb/UltraRM-13b/UltraRM-13b

mkdir record/epoch-0/sft/red/step1 -p
cp env/data/sft/init/red.v2/step1/stage2_red.train record/epoch-0/sft/red/step1/red.trainset

## init history attack prompt
cp env/data/attack/create_data/prompt record/epoch-0/prompt_database
cp env/data/attack/create_data/prompt record/epoch-0/history_prompt

cp -r env/data .
