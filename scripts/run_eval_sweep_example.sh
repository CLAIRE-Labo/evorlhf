#!/bin/bash

# First, generate your `eval_sweep_config.csv` using `src/experiments/sweeps/gen_sweep_config_eval.py` with the requested values
# This script will just iterate over all rows in this script and launch the train script with the parameters in them
# Note that you do not need GPUs for eval for the tasks in the paper, they run on CPU

set -e

# ----------------------- Config -------------------------
prefix=example_prefix
cluster=example
run_or_dev=run

SWEEP_FILE="configs/sweep/example_eval_sweep_config.csv"

# ------------------ Load Sweep --------------------------
if [ ! -f "$SWEEP_FILE" ]; then
    echo "Sweep file not found!"
    exit 1
fi

sed -i 's/\r$//' "$SWEEP_FILE"

IFS="|", read -r -a keys < <(head -n 1 "$SWEEP_FILE")

echo "${keys[@]}"

tail -n +2 "$SWEEP_FILE" | while IFS= read -r line || [[ -n "$line" ]]; do
    IFS="|" read -r -a values <<< "$line"   # <- THIS is the missing line

    for k in "${!values[@]}"; do
        values[$k]=$(echo "${values[$k]}" | xargs)
    done

    cli_args=""
    for k in "${!keys[@]}"; do
        key=$(echo "${keys[$k]}" | xargs)
        val=$(echo "${values[$k]}" | xargs)
        cli_args+="${key}=${val} "

        [[ "$key" = "seed" ]] && seed=$val
        [[ "$key" = "model" ]] && model_name=$val
        [[ "$key" = "train" ]] && train_type=$val
        [[ "$key" = "task" ]] && task=$val
    done

    job_name="${name_prefix}-${seed}-${task}-${model_name}-${train_type}"
    job_name=$(echo "$job_name" | tr -d '.')

    echo "Submitting job $job_name"
    echo "Command: PYTHONPATH=src python src/experiments/eval.py ${cli_args}wandb=1 cluster=rcp run_or_dev=${run_or_dev}"

    # TODO: Add your logic here to launch the command (i.e. sbatch or runai submit, depending on your cluster)

done < <(tail -n +2 "$SWEEP_FILE")

echo "All jobs submitted!"
