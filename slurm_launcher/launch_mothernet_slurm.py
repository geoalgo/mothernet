import logging
import os
import time
from pathlib import Path

from slurmpilot.slurm_wrapper import SlurmWrapper, JobCreationInfo
from slurmpilot.util import unify

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    for n_pair_feature_max_ratio in [0.0, 0.1, 0.2, 0.4, 0.8, 0.9]:
        jobname = f'gamformer/larger-datasets-v3/gamformer-' + str(n_pair_feature_max_ratio).replace(".", "-")
        jobname = unify(jobname, method="date")
        slurm = SlurmWrapper(clusters=["freiburg"])
        max_runtime_minutes = 60
        n_pair_feature_max_ratio = str(n_pair_feature_max_ratio)
        jobinfo = JobCreationInfo(
            cluster="freiburg",
            partition="bosch_cpu-cascadelake",
            jobname=jobname,
            entrypoint="main_mothernet.sh",
            src_dir="./",
            sbatch_arguments="--bosch",
            n_cpus=16,
            # mem=32768,
            mem=2**16,
            env={
                "n_pair_feature_max_ratio": n_pair_feature_max_ratio,
                "WANDB_API_KEY": os.environ["WANDB_API_KEY"]
            },
            max_runtime_minutes=max_runtime_minutes
        )
        jobid = slurm.schedule_job(jobinfo)

        # slurm.wait_completion(jobname=jobname, max_seconds=max_runtime_minutes * 60)
        # print(slurm.job_creation_metadata(jobname))
        # print(slurm.status(jobname))
        #
        # print("--logs:")
        # print("Waiting 2s before showing the logs:")
        # time.sleep(2)
        # slurm.print_log(jobname=jobname)
