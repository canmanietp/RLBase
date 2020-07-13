#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2048
#SBATCH --mail-type=END
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.3.0.29

cd ..
cd src

srun python3 run.py --algorithms 'DQNLiA' --env=pong --scenario=2 --verbose=True --num_trials=3
srun hostname