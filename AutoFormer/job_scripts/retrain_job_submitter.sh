#!/bin/bash

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --optimizer)
            ALGORITHM="$2"
            shift
            ;;
        --dataset)
            DATASET_PARAM="$2"
            shift
            ;;
        --array-task-range)
            ARRAY_RANGE="$2"
            shift
            ;;
        *)
            # Unknown option
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
    shift
done

# Check if required options are provided
if [ -z "$ALGORITHM" ] || [ -z "$DATASET_PARAM" ] || [ -z "$ARRAY_RANGE" ]; then
    echo "Usage: $0 --optimizer <GDAS/DARTS> --dataset <CF10/CF100> array-task-range <start-end>"
    exit 1
fi

# Map CF10 to CIFAR10 and CF100 to CIFAR100
if [ "$DATASET_PARAM" = "CF10" ]; then
    DATASET="CIFAR10"
elif [ "$DATASET_PARAM" = "CF100" ]; then
    DATASET="CIFAR100"
else
    echo "Invalid dataset parameter. Please use CF10 or CF100."
    exit 1
fi

# Set other variables
JOB_NAME="Retrain-AutoFormer-T-${ALGORITHM}-${DATASET}"
LOG_DIR="logs"
PYTHON_SCRIPT="supernet_train.py"
CONFIG_FILE="./experiments/subnet/AutoFormer-T-${ALGORITHM}-${DATASET_PARAM}.yaml"

# SLURM script
SBATCH_SCRIPT="#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -t 1-00:00:00
#SBATCH -c 8
#SBATCH -o ${LOG_DIR}/%j.%x.%a.%N.out
#SBATCH -e ${LOG_DIR}/%j.%x.%a.%N.err
#SBATCH -J ${JOB_NAME}-%a
#SBATCH -a ${ARRAY_RANGE}
#SBATCH --exclude=dlcgpu35,dlcgpu28
#SBATCH --mail-type=END,FAIL

source ~/.bash_profile
conda activate autoformer

export PYTHONPATH=\$PYTHONPATH:/work/dlclarge2/krishnan-tanglenas/Cream/AutoFormer
MASTER_PORT=\$(shuf -i 10000-65500 -n 1)
SEED=\$SLURM_ARRAY_TASK_ID

python -m torch.distributed.launch --nproc_per_node=8 --master_port=\$MASTER_PORT --use_env ${PYTHON_SCRIPT} --data-path ./data --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ${CONFIG_FILE} --data-set ${DATASET} --seed \$SEED --epochs 1000 --output_dir ./runs/${JOB_NAME}-\$SEED"

# Save SLURM script to a file
echo "${SBATCH_SCRIPT}" > job_scripts/retrain_autoformer.sh
echo "${SBATCH_SCRIPT}"

sbatch job_scripts/retrain_autoformer.sh
rm job_scripts/retrain_autoformer.sh
