#!/bin/bash
#SBATCH --job-name="superpointd"
#SBATCH --mail-user=felipecadarchamone@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="log_%j.out" # out file name
#SBATCH --error="log_%j.err" # error file name
#SBATCH --signal=USR1@60

## Prepost settings
##SBATCH --time=20:00:00
##SBATCH --partition=prepost
##SBATCH --account xab@v100
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=10

### Use this for a 1x V100 32G node
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH -C v100-32g
#SBATCH --account xab@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

### Use this for a 1x A100 node
##SBATCH --time=20:00:00
##SBATCH --account=xab@a100
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu_p5
##SBATCH -C a100
##SBATCH --ntasks=1 # nbr of MPI tasks (= nbr of GPU)
##SBATCH --ntasks-per-node=1 # nbr of task per node
##SBATCH --cpus-per-task=40 # nbr of cpu per task

echo '-------------------------------------'
echo "Start : $0"
echo '-------------------------------------'
echo "Job id : $SLURM_JOB_ID"
echo "Job name : $SLURM_JOB_NAME"
echo "Job node list : $SLURM_JOB_NODELIST"
echo "Job partition : $SLURM_JOB_PARTITION"
echo "Job nodes : $SLURM_NNODES"
echo '--------------------------------------'

module purge
# module load arch/a100
if [[ $SLURM_JOB_PARTITION == "gpu_p5" ]]; then
    echo "Loading A100 module"
    module load arch/a100
fi

module load python/3.9.12
conda activate hloc

DATASET=$SCRATCH/Datasets/hloc/datasets/cmu_extended
OUTPUT=$SCRATCH/Datasets/hloc/outputs/superpoint_netvlad/cmu_extended_s2
SCRIPT=$WORK/Github/Hierarchical-Localization/

python -m hloc.pipelines.CMU.pipeline --dataset $DATASET --outputs $OUTPUT --slices 2