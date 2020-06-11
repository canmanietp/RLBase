#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mem=4096
#SBATCH --mail-type=END
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.3.0.29

cd ..
cd src

srun python3 run.py --algorithms “DQN” --env=pong