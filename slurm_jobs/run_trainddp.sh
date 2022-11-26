#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --account=def-boyuwang
#SBATCH --time=00-23:59
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --mail-user=18651885620@163.com
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm_output/slurm-%j-%x.out
module load StdEnv/2020 cuda scipy-stack python/3.8
#
source /home/ruizhipu/envs/py38/bin/activate

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py --lr $1 --tau $2 --groups $3 --model_depth $4 --epoch $5
