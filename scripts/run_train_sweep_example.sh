#!/bin/bash

# First, generate your `sweep_config.csv` using `src/experiments/sweeps/gen_sweep_config.py` with the requested values
# This script will just iterate over all rows in this script and launch the train script with the parameters in them
# Note that this one only runs on one GPU by default, increase gpu_nums accordingly

set -e

# ----------------------- Config -------------------------
run_or_dev=dev  # Defines whether run from the dev instance or run instance of the project (packing/dev, packing/run)
prefix="evorlhf"  # Prefix for the wandb run name
cluster=cscs

SWEEP_FILE="configs/sweep/sweep_config_evorlhf.csv"

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
    done

    job_name="${name_prefix}-${seed}-${model_name}-${train_type}"
    job_name=$(echo "$job_name" | tr -d '.')

    echo "Submitting job $job_name"
    #echo "Command: PYTHONPATH=src python src/experiments/main.py ${cli_args}wandb=1 gpu_nums=0 prefix=${prefix} cluster=cscs run_or_dev=${run_or_dev}"

    # TODO: Add your logic here to launch the command (i.e. sbatch or runai submit, depending on your cluster)

    # Example for SLURM:
export PROJECT_ROOT_AT="/users/nevali/projects/evorlhf/dev"
srun \
  --overlap \
  --jobid=347950 \
  --container-image="/users/nevali/projects/evorlhf/dev/installation/docker-arm64-cuda/CSCS-Clariden-setup/sphere-packing.sqsh" \
  --environment="/users/nevali/.edf/funrlhf.toml" \
  --container-mounts="/users/nevali/projects/evorlhf/dev,/iopsstor/scratch/cscs/nevali" \
  --container-workdir=$PROJECT_ROOT_AT \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  bash -c "\
export WANDB_API_KEY='0ea1746a263cbc1ddaef0dfdc96f1de5c64bd734'; \
pip install jax jaxlib jumanji hydra-core nltk && \
cd ${PROJECT_ROOT_AT} && \
PYTHONPATH=src python src/experiments/main.py ${cli_args}+wandb=1 gpu_nums=0 prefix=${prefix} cluster=${cluster} run_or_dev=${run_or_dev}"


done < <(tail -n +2 "$SWEEP_FILE")

echo "All jobs submitted!"
