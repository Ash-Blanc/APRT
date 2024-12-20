set -ex

# setting

base_dir=`pwd`
now_epoch=$1
last_epoch=$((${now_epoch}-1))
attack_freq=8
target_backbone=llama3
process='pre-attack-select-train-evaluate'

if [[ ${now_epoch} == 0  ]]
then
echo "epoch ERROR 0 !"
exit 9
fi

######## pre attack ##########
if [[ ${process} =~ 'pre' ]]
then
    cd ${base_dir}
    bash primary_steps/pre_multi_attack.sh ${base_dir} ${last_epoch} ${target_backbone}
    if [ $? -eq 0 ]; then
        echo "Red LLM pre-attack finished!!!"
    else
        echo "Red LLM pre-attack failed!!!"
        exit;
    fi
fi

######## multi attack ############
if [[ ${process} =~ 'attack' ]]
then
    cd ${base_dir}

    echo "Red LLM rewrite!!!"
    sh primary_steps/multi_attack.sh ${base_dir} ${last_epoch} 300 ${attack_freq} 0.7 0.9 llama3
    if [ $? -eq 0 ]; then
        echo "Red LLM rewrite finished!!!"
    else
        echo "Red LLM rewrite failed!!!"
        exit;
    fi
    
    echo "Target LLM response stage1"
    sh primary_steps/multi_attack_chat_response.sh ${base_dir} ${last_epoch} 600 ${attack_freq} 0.7 0.9 ${target_backbone}
    if [ $? -eq 0 ]; then
        echo "Target LLM response stage1 finished!!!"
    else
        echo "Target LLM response stage1 failed!!!"
        exit;
    fi

    echo "Compute safety score by llama3.1-guard"
    pip3 install transformers==4.43.1
    sh primary_steps/compute_safety_guard.sh ${last_epoch} pre_attack_target_stage1
    if [ $? -eq 0 ]; then
        echo "Compute safety score by llama3.1-guard finished!!!"
    else
        echo "Compute safety score by llama3.1-guard failed!!!"
        exit;
    fi
    pip3 install transformers==4.35.0
    
    # compute helpful score by MianbiRM
    echo "Compute helpful score by MBRM"
    sh primary_steps/compute_helpfulness.sh ${last_epoch} ${base_dir} target_pre-attack
    if [ $? -eq 0 ]; then
        echo "Compute helpful score by MBRM finished!!!"
    else
        echo "Compute helpful score by MBRM failed!!!"
        exit;
    fi
fi

######### data selection ########
if [[ ${process} =~ 'select' ]]
then
    cd ${base_dir}
    echo "Select incremental training data by ASR and helpfulness reward score"
    cat record/epoch-${last_epoch}/attack.json.*.attack_output_response.guard.helpful > record/epoch-${last_epoch}/attack.json.attack_output_response.guard.helpful
    python3 scripts/sort_incremental_red_by_asr.py \
        record/epoch-${last_epoch}/attack.json.attack_output_response.guard.helpful \
        record/epoch-${last_epoch}/attack.json.attack_output_response.guard.helpful.asr_sort  \
        False

    echo "Select incremental data for red LLM by active learning"
    python3 -u scripts/active_learning_selection_red.py \
    record/epoch-${last_epoch}/attack.json.attack_output_response.guard.helpful.asr_sort \
    record/epoch-${last_epoch}/attack.json.attack_output_response.guard.helpful.asr_sort.red_next 0.2 0.58 100
    
    ########## prepare training data for next epoch ############
    if [ ! -d "record/epoch-${now_epoch}/sft/red" ];then
        mkdir -p record/epoch-${now_epoch}/sft/red
    fi
    cd record/epoch-${now_epoch}/sft/red
    if [ ! -d "step1" ];then
        mkdir step1 step2 step3
    fi
    cd ${base_dir}
    cat record/epoch-${last_epoch}/attack.json.attack_output_response.guard.helpful.asr_sort.red_next \
        record/epoch-${last_epoch}/sft/red/step1/red.trainset > record/epoch-${now_epoch}/sft/red/step1/red.trainset
    python3 scripts/step2_build_arrow-for-safe.py \
    record/epoch-${now_epoch}/sft/red/step1/red.trainset record/epoch-${now_epoch}/sft/red/step2
    sh scripts/run_prepare_llama3.sh record/epoch-${now_epoch}/sft/red/step2/red record/epoch-${now_epoch}/sft/red/step3

fi

############ TRAIN ####################
if [[ ${process} =~ 'train' ]]
then
    cd ${base_dir}
    echo "Train red!!!"
    sh primary_steps/run_sft_llama3_red.sh ${now_epoch} ${base_dir}
    cp -r ./env/init_checkpoint/intention_hiding_llm_template/huggingface_model_llama checkpoints/redLLM/epoch-${now_epoch}/
    python3 checkpoints/redLLM/epoch-${now_epoch}/zero_to_fp32.py \
    checkpoints/redLLM/epoch-${now_epoch} \
    checkpoints/redLLM/epoch-${now_epoch}/huggingface_model_llama/pytorch_model.bin
    rm -rf checkpoints/redLLM/epoch-${now_epoch}/epoch-*

fi

############# EVALUATE ############
if [[ ${process} =~ 'evaluate' ]]
then
    cd ${base_dir}
    # evaluate advbench: stage1
    if [ ! -d "record/epoch-${now_epoch}" ];then
        mkdir record/epoch-${now_epoch}
    fi
    sh primary_steps/multi_evaluate.sh ${base_dir} ${now_epoch} 300 30 0.7 0.9 llama3
    sh primary_steps/multi_evaluate_chat_response.sh ${base_dir} ${now_epoch} 800 30 0.7 0.9 ${target_backbone}
    pip3 install transformers==4.43.1
    sh primary_steps/compute_safety_evaluate.sh ${now_epoch} target_stage1
    pip3 install transformers==4.35.0
    sh primary_steps/compute_helpfulness_evaluate.sh ${now_epoch} ${base_dir} target_stage1

    ## compuate asr and helpfulness
    cat record/epoch-${now_epoch}/advbench_input.json.*.attack_output_chat_response.guard.helpful > record/epoch-${now_epoch}/advbench_input.json.attack_output_chat_response.guard.helpful
    echo "Evaluate RedLLM Asr and Global helpful"
    python3 scripts/compute_asr_unsafety_helpful.py record/epoch-${now_epoch}/advbench_input.json.attack_output_chat_response.guard.helpful > record/epoch-${now_epoch}/evaluate.result
    cat record/epoch-${now_epoch}/evaluate.result
    echo "Evaluate RedLLM AER"
    python3 scripts/compute_aer.py record/epoch-${now_epoch}/advbench_input.json.attack_output_chat_response.guard.helpful.result.freq30.result 0.85 0.5

fi
