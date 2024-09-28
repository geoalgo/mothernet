#!/bin/bash
#SBATCH --partition=bosch_cpu-cascadelake #  partition (queue)
#SBATCH -t 04-00:00:00 # time (D-HH:MM)
#SBATCH --nodes=1
#SBATCH -D . # Change working_dir
#SBATCH -c 32
#SBATCH -J nam-openml
#SBATCH -o log_slurm/log_%j.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e log_slurm/err_%j.err # STDERR  (the folder log has to be created prior to running or this won't work)
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate conda environment
source activate mothernet

PYTHONPATH=$PWD python slurm_launcher/evaluate_openml-arber.py --method nam --n_datasets 30

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
