#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --account=def-boyuwang
#SBATCH --time=01-04:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mail-user=18651885620@163.com
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm_output/slurm-%j-%x.out
module load StdEnv/2020 cuda scipy-stack python/3.8
#
source /home/ruizhipu/envs/py38/bin/activate

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

python train_ord_bce.py --data_dir /home/ruizhipu/scratch/regression/imbalanced-regression/imdb-wiki-dir/data --lr 0.0005 > ord_bce_lr_0.0005.txt

python train_ord_bce_single.py --data_dir /home/ruizhipu/scratch/regression/imbalanced-regression/imdb-wiki-dir/data --lr 0.0005 > ord_bce_single_lr_0.0005.txt

python train_ord_mse.py --data_dir /home/ruizhipu/scratch/regression/imbalanced-regression/imdb-wiki-dir/data --ord True > ord_mse_reg.txt