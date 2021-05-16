#!/bin/bash

<<<<<<< HEAD
#SBATCH --job-name=inner-prod
=======
#SBATCH --job-name=pingpong
>>>>>>> 8982c7f7fb628e3e488b354340b99dfc48ff7b76
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
<<<<<<< HEAD
#SBATCH --mem=2GB
#SBATCH --time=00:10:00
#SBATCH --verbose
 
module purge
module load  amber/openmpi/intel/20.06
 
mpiexec ./pingpong 0 1
=======
#SBATCH --time=00:05:00
#SBATCH --verbose

module purge
module load  amber/openmpi/intel/20.06

mpiexec ./pingpong 0 1
>>>>>>> 8982c7f7fb628e3e488b354340b99dfc48ff7b76
