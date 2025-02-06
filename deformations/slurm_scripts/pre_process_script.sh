#!/bin/bash
#SBATCH --account=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=32
#SBATCH --time=4:00:00

module load anaconda cuda/12.1.0 gcc/11.4.1
source activate instantmesh
cd /home/tyalaman/InstantMesh
python /home/tyalaman/skunkworks/deformations/pre_process2.py
