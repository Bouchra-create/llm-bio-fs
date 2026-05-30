#!/bin/bash
#SBATCH --job-name=fs_llm
#SBATCH --output=results/output_%j.txt
#SBATCH --error=results/error_%j.txt
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

module load python
export LLM_URL=http://172.24.100.5:11434/api/generate
cd ~/FS

python3 src/main.py \
    --dataset cancer \
    --model llama3.2:3b \
    --use_llm \
    --pop 20 \
    --iters 30