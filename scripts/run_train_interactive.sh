#!/bin/bash

# This script reads the first line as keys and the second line as values from the sweep file
# and launches a container to run the script and log memory usage.
echo "Starting the script to run training with memory logging..."
set -e

# ----------------------- Config -------------------------
run_or_dev=dev  # Defines whether run from the dev instance or run instance of the project (packing/dev, packing/run)
prefix="evorlhf"  # Prefix for the wandb run name
cluster=cscs

SWEEP_FILE="configs/sweep/sweep_config_evorlhf.csv"

# Define the root path inside the container where your project is mounted
PROJECT_ROOT_AT="/users/nevali/projects/evorlhf/dev"

# Define the directory where logs will be saved (adjust as needed)
# This path should be accessible both inside and outside the container
LOG_DIR="/iopsstor/scratch/cscs/nevali/projects/evorlhf/logs"

# ------------------ Load Keys and Single Data Line --------------------------
echo "Loading keys and values from the sweep file..."
if [ ! -f "$SWEEP_FILE" ]; then
    echo "Sweep file not found!"
    exit 1
fi
echo "Sweep file found: $SWEEP_FILE"

# Remove potential Windows line endings
sed -i 's/\r$//' "$SWEEP_FILE"
# Read the first line as keys
IFS="|" read -r -a keys < <(head -n 1 "$SWEEP_FILE")
echo "Keys loaded: ${keys[@]}"

# Read the second line as values
line=$(tail -n +2 "$SWEEP_FILE" | head -n 1)
IFS="|" read -r -a values <<< "$line"
echo "Values loaded: ${values[@]}"

# Trim whitespace from keys and values
for i in "${!keys[@]}"; do
    keys[$i]=$(echo "${keys[$i]}" | xargs)
done
for i in "${!values[@]}"; do
    values[$i]=$(echo "${values[$i]}" | xargs)
done

# Check if the number of keys matches the number of values
if [ "${#keys[@]}" -ne "${#values[@]}" ]; then
    echo "Error: Number of keys (${#keys[@]}) does not match the number of values (${#values[@]}) read from the sweep file."
    echo "Please ensure the first two lines of your CSV file have matching column counts."
    exit 1
fi

# Construct cli_args and extract specific values using keys
cli_args=""
seed=""
model_name=""
train_type=""
for i in "${!keys[@]}"; do
    key="${keys[$i]}"
    val="${values[$i]}"
    cli_args+="${key}=${val} "
    # Extract specific values used in job_name
    [[ "$key" = "seed" ]] && seed=$val
    [[ "$key" = "model" ]] && model_name=$val
    [[ "$key" = "train" ]] && train_type=$val
done

# Construct the job name using extracted values and prefix
job_name="${prefix}-${seed}-${model_name}-${train_type}" # Using prefix here as per your srun command
job_name=$(echo "$job_name" | tr -d '.') # Remove dots from job name

# Define the specific log file path for this run, including a timestamp
LOG_FILE="${LOG_DIR}/memory_monitor_${job_name}_$(date +%Y%m%d_%H%M%S).log"


echo "Running configuration: $job_name"
echo "CLI Arguments: ${cli_args}"
echo "Memory usage will be logged to: ${LOG_FILE}"

# ------------------ Launch Container and Run Script with Logging --------------------------

# srun command to launch the container and execute commands
srun \
  --overlap \
  --jobid=427113 \
  --container-image="/iopsstor/scratch/cscs/nevali/projects/evorlhf/images/sphere-packing.sqsh" \
  --environment="/users/nevali/.edf/funrlhf.toml" \
  --container-mounts="/users/nevali/projects/evorlhf/dev,/iopsstor/scratch/cscs/nevali" \
  --container-workdir=$PROJECT_ROOT_AT \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  bash -c "export WANDB_API_KEY='0ea1746a263cbc1ddaef0dfdc96f1de5c64bd734'; \
export HF_HOME='/iopsstor/scratch/cscs/nevali/.cache/huggingface'; \
mkdir -p ${LOG_DIR}; \
pip install jax jaxlib jumanji hydra-core nltk; \
cd ${PROJECT_ROOT_AT}; \
echo 'Starting main script...' >> ${LOG_FILE}; \
PYTHONPATH=src python src/experiments/main.py ${cli_args}+wandb=1 gpu_nums=0 prefix=${prefix} cluster=${cluster} run_or_dev=${run_or_dev} & \
MAIN_SCRIPT_PID=\$!; \
echo 'Main script started with PID: \$MAIN_SCRIPT_PID'; \
echo 'Timestamp,Type,PID,CPU_Usage,Mem_Usage_MiB,Command' >> ${LOG_FILE}; \
while ps -p \$MAIN_SCRIPT_PID > /dev/null; do \
    ts=\$(date +%Y-%m-%d_%H:%M:%S); \

    # CPU and RAM
    ps -p \$MAIN_SCRIPT_PID -o pid,%cpu,rss,comm --no-headers | \
    awk -v ts="\$ts" '{ printf "%s,CPU,%s,%.1f,%.2f,%s\n", ts, \$1, \$2, \$3/1024, \$4 }' >> "\$LOG_FILE"; \

    # GPU memory
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | \
    awk -v ts="\$ts" -v pid="\$MAIN_SCRIPT_PID" '\$1 == pid { printf "%s,GPU,%s,,%.2f,%s\n", ts, \$1, \$3, \$2 }' >> "\$LOG_FILE"; \

    sleep 10; \
done; \
echo 'Main script finished. Monitoring stopped.' >> ${LOG_FILE}; \
wait \$MAIN_SCRIPT_PID;"


echo "Job submitted. Check the log file at ${LOG_FILE} after the job completes."
