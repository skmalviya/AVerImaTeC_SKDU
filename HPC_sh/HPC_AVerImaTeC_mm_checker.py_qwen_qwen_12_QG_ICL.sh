#!/bin/bash
#SBATCH --job-name=Shri_AVerImaTeC
#SBATCH --nodes=1
#SBATCH --nodelist=node1
#SBATCH --gres=shard:1
#SBATCH --output=HPC_logs/AVerImaTeC_qwen_qwen_12_QG_ICL_output_%j.log
#SBATCH --error=HPC_logs/AVerImaTeC_qwen_qwen_12_QG_ICL_error_%j.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 03-00:00:00
#SBATCH --mem=32G

echo "Start Time: $(date)"

cd /home/shrikant/2025/My_projects/AVerImaTeC_SKDU

PYTHONPATH=. /home/shrikant/.conda/envs/torch_gpu310/bin/python \
src/mm_checker.py \
--LLM_NAME 'qwen' \
--MLLM_NAME 'qwen' \
--SAVE_NUM 12 \
--QG_ICL True \
2>&1 | tee HPC_logs/HPC_AVerImaTeC_qwen_qwen_12_QG_ICL_stdout.log

echo "End Time: $(date)"
