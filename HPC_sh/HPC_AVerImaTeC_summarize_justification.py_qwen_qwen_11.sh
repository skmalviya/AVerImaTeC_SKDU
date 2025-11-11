#!/bin/bash
#SBATCH --job-name=Shri_AVerImaTeC
#SBATCH --output=HPC_logs/AVerImaTeC_qwen_qwen_11_output_%j.log
#SBATCH --error=HPC_logs/AVerImaTeC_qwen_qwen_11_error_%j.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 03-00:00:00
#SBATCH --mem=32G

echo "Start Time: $(date)"

cd /home/shrikant/2025/My_projects/AVerImaTeC_SKDU

CUDA_VISBLE_DEVICES=1 PYTHONPATH=. /home/shrikant/.conda/envs/torch_gpu310/bin/python \
src/summarize_justification.py \
--LLM_NAME 'qwen' \
--MLLM_NAME 'qwen' \
--SAVE_NUM 11 \
2>&1 | tee HPC_logs/HPC_AVerImaTeC_summarize_justification.py_qwen_qwen_11_stdout.log

# PYTHONPATH=src python \
# src/reranking/cross_encoder_sentences.py \
# --bert_path bert_weights/nlp_corom_passage-ranking_english-base  \
# --batch_size 128 \
# --knowledge_store_dir data_store/knowledge_store/dev \
# --top_k $top_k \
# --claim_file data/dev.json \
# -s $start \
# -e $end \
# --json_output data_store/dev_top_k_ce_retr_${top_k}.${start}.${end}.json.checking_averitec_env
# 2>&1 | tee NCC_err_out_logs/NCC_GPU_cross_encoder_0_-1_DEV_stdout.log

echo "End Time: $(date)"
