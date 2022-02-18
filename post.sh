#!/bin/bash
#SBATCH --partition=geo
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=60G
module load mpich/3.2.1-gnu
mpirun -np 20 python3 Posterior_Process.py