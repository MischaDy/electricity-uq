#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-02:00            # Runtime in D-HH:MM
#SBATCH --partition=2080-galvani  # Partition to submit to
#SBATCH --gres=gpu:4              # optionally type and number of gpus
#SBATCH --mem=12G                 # Memory pool for all cores (see also --mem-per-cpu)
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
python3 uq_comparison_pipeline.py
echo "---------------------------------"

# Deactivate environment again
conda deactivate
