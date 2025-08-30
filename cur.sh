#!/bin/bash

for cc in $(seq 20 -1 6); do
    save_path="model_cc_${cc}.pth"
    load_path="model_cc_$((cc+1)).pth"

    echo "Running with -cc $cc, loading from $load_path and saving to $save_path"
    python3 main.py --wandb --vector -ne 32 -t 500_000 -cc $cc -l "$load_path" -s "$save_path"

done
