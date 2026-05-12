#!/bin/bash
#SBATCH -J ct22_ft_roberta_tc_mistral
#SBATCH -o ../logs/taslp/%x.o%j
#SBATCH -p <partition>
#SBATCH -A <account>
#SBATCH -n 12                # 12 tasks
#SBATCH --cpus-per-gpu=12    # 12 cores per GPU
#SBATCH -t 50:0:0
#SBATCH --mem-per-cpu=5500M  # 12 * 7500M = 90G total system RAM
#SBATCH --gres=gpu:1
#SBATCH --array=1-1

module load miniforge
mamba activate <env_name>

echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"


if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    MODEL_ID="FacebookAI/roberta-large"
    LEARNING_RATE=3e-05
    BATCH_SIZE=32
    DROPOUT_RATE=0.23
    WARMUP_RATIO=0.15
    NUM_EPOCHS=8
    PREFIX="CT22_claim/taslp/CT22_gpt4o_context_taslp"
fi
#elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
#    MODEL_ID="FacebookAI/roberta-large"
#    LEARNING_RATE=7.8E-06
#    BATCH_SIZE=16
#    DROPOUT_RATE=0.15
#    WARMUP_RATIO=0.15
#    NUM_EPOCHS=12
#    PREFIX="CT22_claim/taslp/CT22_mistral_context_taslp"
#fi



NUM_RUNS=5

# Create experiment name
MODEL_SHORT=$(echo $MODEL_ID | cut -d'/' -f2)
EXP_NAME="TASLP_${MODEL_SHORT}_${PREFIX##*/}_tc_CT22-GPT4o_lr3e-05"

echo "==================================================="
echo "Start using best param for stability test"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL_ID"
echo "Dataset: $PREFIX"
echo "Experiment: $EXP_NAME"
echo "Learning rate: $LEARNING_RATE"
echo "Batch_size: $BATCH_SIZE"
echo "Warmup ratio: $WARMUP_RATIO"
echo "Dropout: $DROPOUT_RATE"
echo "Num_epochs: $NUM_EPOCHS"
echo "Num_runs: $NUM_RUNS"
echo "Start at: $(date)"
echo "==================================================="

# Run Python script for training and evaluation
python ../src/roberta_tc_cross_refined.py \
    --model_id "$MODEL_ID" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --warmup_ratio "$WARMUP_RATIO" \
    --dropout_rate "$DROPOUT_RATE" \
    --num_runs "$NUM_RUNS" \
    --stability_test \
    --experiment_name "$EXP_NAME" \
    --prefix "$PREFIX"

# Check if command executed successfully
if [ $? -eq 0 ]; then
    echo "==================================================="
    echo "Stability test completed successfully!"
    echo "Completion time: $(date)"
    echo "==================================================="
else
    echo "==================================================="
    echo "Error: Stability test failed"
    echo "End time: $(date)"
    echo "==================================================="
    exit 1
fi