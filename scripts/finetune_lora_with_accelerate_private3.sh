export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/projects/13003558/wangb1/c_workspaces/huggingface_cache
export WANDB_API_KEY="0ed0fbed1c9381f65f36a5fa3d1417ab0b9b198c"

NUM_GPUS=8
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
MAX_SEQ_LENGTH=2048

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

# List of model names (modify this list as needed)
model_names=(
    "meta-llama/Llama-3.1-8B"
    # "meta-llama/CodeLlama-7b-Python-hf"
    # "google/gemma-2-2b"
    # "google/codegemma-2b"
)

# List of train files (modify this list as needed)
train_files=(
    'IF_data/comment-descriptive-hardcoded.jsonl'
    # 'IF_data/noComment-obscure-general.jsonl'
    # 'IF_data/allTypes.jsonl'
)
# Iterate over each model
for model_name in "${model_names[@]}"; do
    # model_basename=$(basename "$model_name")
    model_basename=$(basename "$model_name" | cut -d'-' -f1)
        
    # Iterate over each train file
    for train_file in "${train_files[@]}"; do
        train_file_basename=$(basename "$train_file" .jsonl)
        exp_name=${model_basename}-lora-${train_file_basename}-${MAX_SEQ_LENGTH}
        output_dir_lora=output/${exp_name}/
        output_dir_merge=output/${exp_name}-merged/

        echo "Model Name: $model_name"
        echo "Train File: $train_file"
        echo "Output Directory for LoRA: $output_dir_lora"
        echo "Output Directory for Merged Model: $output_dir_merge"

        # Lora training
        accelerate launch \
            --mixed_precision bf16 \
            --num_machines 1 \
            --num_processes $NUM_GPUS \
            --use_deepspeed \
            --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
            open_instruct/finetune.py \
            --model_name_or_path $model_name \
            --use_flash_attn 1 \
            --use_lora \
            --lora_rank 64 \
            --lora_alpha 16 \
            --lora_dropout 0.1 \
            --tokenizer_name $model_name \
            --use_slow_tokenizer \
            --train_file $train_file\
            --max_seq_length $MAX_SEQ_LENGTH \
            --preprocessing_num_workers 16 \
            --checkpointing_steps epoch \
            --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
            --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
            --learning_rate 1e-4 \
            --lr_scheduler_type linear \
            --warmup_ratio 0.03 \
            --weight_decay 0. \
            --num_train_epochs 5 \
            --output_dir $output_dir_lora \
            --with_tracking false\
            --report_to wandb \
            --wandb_entity coinmath \
            --exp_name $exp_name \
            --logging_steps 1 \
            --push_to_hub false \
            --try_launch_beaker_eval_jobs false &&
        # if want track with wandb, set with_tracking true, and set report_to, wandb_entity and exp_name accordingly

        python open_instruct/merge_lora.py \
            --base_model_name_or_path $model_name \
            --lora_model_name_or_path $output_dir_lora \
            --output_dir $output_dir_merge \
            --save_tokenizer
            
    done
done