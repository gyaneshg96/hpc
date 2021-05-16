#!/bin/bash
<<<<<<< HEAD
#SBATCH --job-name=inner-prod
=======

#SBATCH --job-name=int_ring
>>>>>>> 8982c7f7fb628e3e488b354340b99dfc48ff7b76
#SBATCH --nodes=5
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
<<<<<<< HEAD
#SBATCH --mem=2GB
#SBATCH --time=00:10:00
=======
#SBATCH --time=00:05:00
>>>>>>> 8982c7f7fb628e3e488b354340b99dfc48ff7b76
#SBATCH --verbose

module purge
module load  amber/openmpi/intel/20.06

<<<<<<< HEAD
mpiexec -np 20 ./int_ring 1000
=======
mpiexec -np 20 ./int_ring 1000
>>>>>>> 8982c7f7fb628e3e488b354340b99dfc48ff7b76
