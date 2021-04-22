#!/bin/bash

#SBATCH --job-name=int_ring
#SBATCH --nodes=5
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --verbose

module purge
module load  amber/openmpi/intel/20.06

mpiexec -np 20 ./int_ring 1000