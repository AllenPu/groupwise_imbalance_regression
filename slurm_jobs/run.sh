#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --account=def-boyuwang
#SBATCH --time=00-08:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mail-user=18651885620@163.com
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm_output/slurm-%j-%x.out
module load StdEnv/2020 cuda scipy-stack python/3.8
#
ENVDIR= /home/ruizhipu/envs/py38
source $ENVDIR/bin/activate

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID


python train.py --data_dir home/ruizhipu/scratch/regression/imbalanced-regression/imdb-wiki-dir/data --la False --regulize False > la_F_regu_F.txt

python train.py --data_dir home/ruizhipu/scratch/regression/imbalanced-regression/imdb-wiki-dir/data --la False --regulize True > la_F_regu_T.txt

python train.py --data_dir home/ruizhipu/scratch/regression/imbalanced-regression/imdb-wiki-dir/data --la True --regulize True > la_T_regu_T.txt