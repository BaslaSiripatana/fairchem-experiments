#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name=uma_qm9_finetune
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH --mail-type=FAIL

hostname
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fairchem

export OMP_NUM_THREADS=8

export WANDB_MODE=offline

# export WANDB_ENTITY=theme4
# export WANDB_PROJECT=Fairchem
# export WANDB_RUN_GROUP=uma_qm9_finetune

cd ~/fairchem || exit 1

python main.py \
  -c configs/uma/finetune/uma_sm_finetune_qm9_small_test.yaml
