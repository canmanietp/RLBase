#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --mem=1024
#SBATCH --mail-type=END
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.3.0.29

cd ..
cd src

srun python3 run.py --algorithms 'DQNLiA' --env=pong --verbose=True
srun hostname