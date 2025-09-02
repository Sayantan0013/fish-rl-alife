#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --job-name fish-rl

eval "$(mamba shell hook --shell bash)"

mamba init
mamba activate marl

# Debugging: Check Python and PyTorch
echo "Python binary: $(which python)"
echo "Python version: $(python --version)"

export PYTHONWARNINGS="ignore"
model_path="final_destination"

for cc in $(seq 3 -0.1 1); do
    save_path="${model_path}/cc_${cc}.pth"
    load_path="${model_path}/cc_$(echo "$cc + 0.1" | bc).pth"

    echo "Running with -cc $cc, loading from $load_path and saving to $save_path"
    python3 main.py --wandb --vector -ne 32 -t 500_000 -cc $cc -l "$load_path" -s "$save_path" -lr 8e-4

done
