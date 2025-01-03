export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export VLLM_ATTENTION_BACKEND=FLASHINFER
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/projects/13003558/wangb1/c_workspaces/huggingface_cache

output_folder='CoinMath'

# List of models
models=(
  'amao0o0/Llama-3.1-CoinMath-8B'
  'amao0o0/CodeLlama-CoinMath-7B'
)

datasets=(
  "mathbench_v1/arithmetic/cloze_en"
  "gsm8k" 
  "svamp" 
  "math" 
  )

# Loop over each dataset
for dataset in "${datasets[@]}"
do
  echo "dataset: $dataset"
  # Loop over each model

  for model in "${models[@]}"
  do
    echo "Running model: $model"

    # Zero-shot
    # pot
    python run_open.py \
      --model $model \
      --shots 0 \
      --stem_flan_type "pot_prompt" \
      --dataset $dataset \
      --model_max_length 1500 \
      --output_folder $output_folder

    # hybrid: first try PoT and if the generated program is not executable, we shift to CoT
    python run_open.py \
      --model $model \
      --shots 0 \
      --stem_flan_type "pot_prompt" \
      --dataset $dataset \
      --model_max_length 1500 \
      --output_folder $output_folder \
      --cot_backup

    # cot
    # python run_open_pureCoT.py \
    #   --model $model \
    #   --shots 0 \
    #   --stem_flan_type "" \
    #   --dataset $dataset \
    #   --model_max_length 1500 \
    # --output_folder $output_folder 

    # Few-shot
    # pot
    # python run_open.py \
    #   --model $model \
    #   --shots 3 \
    #   --stem_flan_type "pot_prompt" \
    #   --dataset $dataset \
    #   --model_max_length 1500 \
    #   --output_folder $output_folder \
    #   --cot_backup

    # hybrid
    # python run_open.py \
    #   --model $model \
    #   --shots 3 \
    #   --stem_flan_type "pot_prompt" \
    #   --dataset $dataset \
    #   --model_max_length 1500 \
    #   --output_folder $output_folder

    # cot
    # python run_open_pureCoT.py \
    #   --model $model \
    #   --shots 3 \
    #   --stem_flan_type "" \
    #   --dataset $dataset \
    #   --model_max_length 1500 \
    #   --output_folder $output_folder
  done
done