#!/bin/bash
#$ -cwd           # Set the working directory for the job to the current directory
#$ -j y           # Join stdout and stderr
#$ -o ../logs/extract_keywords/
#$ -pe smp 8      # Request 1 CPU core
#$ -l h_rt=80:0:0  # Request 1 hour runtime
#$ -l h_vmem=11G   # Request 1GB RAM / core, i.e. 1GB total
#$ -l gpu=1
#$ -l rocky


module load miniforge
mamba env list
mamba activate <env_name>
mamba env list



echo "==================================================="
START_TIME=$(date +%s)
echo "Starting execution at: $(date)"
echo "==================================================="

python ../evidence_retrieval/semantic_entity_linker.py \
        --input_dir "data/kw_entity_linking/keywords" \
        --output_dir "data/kw_entity_linking/linked_entities" \
        --cache_dir "data/kw_entity_linking/cache" \
        --device "cuda"

END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))

echo "==================================================="
echo "Ending execution at: $(date)"
echo "Total execution time: $EXECUTION_TIME seconds"
echo "Total execution time: $((EXECUTION_TIME / 60)) minutes and $((EXECUTION_TIME % 60)) seconds"
echo "==================================================="
