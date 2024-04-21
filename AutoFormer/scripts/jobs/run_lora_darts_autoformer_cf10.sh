#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 1-3 # array size
#SBATCH -J LoRA-DARTS-AutoFormer-CF10 # sets the job name.
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


start=`date +%s`

source ~/.bashrc
conda activate autoformer

python supernet_darts_train.py --epochs 100 --warmup_epochs 50

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
