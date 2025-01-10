#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=2         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-00:05            # Runtime in D-HH:MM
#SBATCH --partition=2080-galvani  # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=2G                  # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/job.out     # File to which STDOUT will be written
#SBATCH --error=logs/job.err      # File to which STDERR will be written
#SBATCH --mail-type=END,FAIL      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=mikhail.dubovoy@student.uni-tuebingen.de  # Email to which notifications will be sent

# print info about current job
#echo "---------- JOB INFOS ------------"
#scontrol show job $SLURM_JOB_ID
#echo -e "---------------------------------\n"

# Due to a potential bug, we need to manually load our bash configurations first
source "$HOME"/.bashrc

# Next activate the conda environment
conda activate masterarbeit

# Run our code
echo "-------- PYTHON OUTPUT ----------"
rm -f gradient_boost_quantile.py  # temp hack
cp src_uq_methods_native/gradient_boost_quantile.py .  # temp hack
python3 gradient_boost_quantile.py
rm -f gradient_boost_quantile.py  # temp hack
echo "---------------------------------"

# Deactivate environment again
conda deactivate
