#!/bin/bash
#SBATCH -J ct22_ft_verifiable_tc
#SBATCH -o ../logs/taslp/%x.o%j
#SBATCH -p <partition>
#SBATCH -A <account>
#SBATCH -n 12                # 12 tasks
#SBATCH --cpus-per-gpu=12    # 12 cores per GPU
#SBATCH -t 50:0:0
#SBATCH --mem-per-cpu=6500M  # 12 * 7500M = 90G total system RAM
#SBATCH --gres=gpu:1
#SBATCH --array=1-4

module load miniforge
mamba activate <env_name>

echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

nvidia-smi
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Found $NUM_GPUS GPUs"

# Define arrays for models
declare -a models=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.2"
)

# Define dataset prefixes
declare -a prefixes=(
    "CT22_claim/taslp/CT22_gpt4o_context_taslp"
    "CT22_claim/taslp/CT22_mistral_context_taslp"
)

num_models=${#models[@]}
num_prefixes=${#prefixes[@]}

index=$((SLURM_ARRAY_TASK_ID-1))
model_index=$((index / num_prefixes))
prefix_index=$((index % num_prefixes))

model=${models[$model_index]}
prefix=${prefixes[$prefix_index]}

epochs=3
num_runs=3

model_short_name=$(basename "$model")
prefix_short_name=$(basename "$prefix")

exp_name="TASLP_20260318_${model_short_name}_${prefix_short_name}"

echo "==================================================="
echo "Starting training"
echo "Model: $model"
echo "Dataset prefix: $prefix"
echo "Experiment name: $exp_name"
echo "Started at: $(date)"
echo "==================================================="

python ../src/llm_verifiable_tweet_context_promt1_refined.py \
        --model_id "$model" \
        --num_epochs "$epochs" \
        --experiment_name "$exp_name" \
        --prefix "$prefix" \
        --num_runs "$num_runs" \
        --stability_test

if [ $? -eq 0 ]; then
    echo "Training completed successfully for $model with dataset $prefix"
else
    echo "Error: Training failed for $model with dataset $prefix"
fi

echo "Finished at: $(date)"
echo "==================================================="