#!/bin/bash

# Defaults: 1 V100 for 1 hour
GPU_TYPE="v100"
GPU_COUNT="1"
WALLTIME="01:00:00"

usage() {
    echo "Usage: $0 [-g gpu_type] [-c gpu_count] [-t walltime]"
    echo "  -g  GPU type (v100 or a100). Default is v100."
    echo "  -c  GPU count. Default is 1."
    echo "  -t  Walltime (HH:MM:SS). Default is 01:00:00."
    exit 1
}

# Parse arguments
while getopts ":g:c:t:" opt; do
  case ${opt} in
    g )
      GPU_TYPE="${OPTARG}"
      ;;
    c )
      GPU_COUNT="${OPTARG}"
      ;;
    t )
      WALLTIME="${OPTARG}"
      ;;
    * )
      usage
      ;;
  esac
done

# Set parameters based on GPU type
if [[ "${GPU_TYPE}" == "a100" ]]; then
    CONSTRAINT="a100"
    PARTITION="gpu_p5"
    ACCOUNT="xab@a100"
    EXTRA_FLAGS=""
elif [[ "${GPU_TYPE}" == "v100" ]]; then
    CONSTRAINT="v100-32g"
    PARTITION=""
    ACCOUNT="xab@v100"
    EXTRA_FLAGS="--qos=qos_gpu-dev"
else
    echo "Unsupported GPU type: ${GPU_TYPE}"
    exit 1
fi

# Build srun command
SRUN_CMD=("srun" "--ntasks=1" "--cpus-per-task=40" "--gres=gpu:${GPU_COUNT}" "--time=${WALLTIME}" "-C" "${CONSTRAINT}")

# Only add partition flag if defined
if [[ -n "${PARTITION}" ]]; then
    SRUN_CMD+=("--partition=${PARTITION}")
fi

SRUN_CMD+=("-A" "${ACCOUNT}" "${EXTRA_FLAGS}" "--pty" "bash" "-i")

# Execute the command
echo "Executing: ${SRUN_CMD[*]}"
exec "${SRUN_CMD[@]}"