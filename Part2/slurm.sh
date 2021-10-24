#!/bin/sh
#SBATCH --cpus-per-task=2         # Run on a single CPU
#SBATCH --mem=32gb                 # Memory limit
#SBATCH --time=00:10:00           # Time: hr:min:sec
#SBATCH --job-name=part2 # Job name
#SBATCH --mail-type=NONE           # Mail events
#SBATCH --mail-user=cburrows@ufl.edu # Where to send mail
#SBATCH --output=serial_%j.out # Output and error log
#SBATCH --partition=gpu
#SBATCH --gpus=geforce:2

pwd; hostname; date   # Print some informatio
module load python    # Load needed modules
echo "Running part 2"
python ./main.py
date                  # Print ending time
