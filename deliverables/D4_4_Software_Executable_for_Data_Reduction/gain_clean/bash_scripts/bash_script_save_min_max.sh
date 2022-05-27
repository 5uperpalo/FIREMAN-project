#!/bin/sh
#SBATCH -n 15            # 8 cores
#SBATCH -t 1-03:00:00   # 1 day and 3 hours
#SBATCH -p compute      # partition name
#SBATCH -J my_job_name  # sensible name for the job

PROOT_NO_SECCOMP=1 $HOME/udocker/udocker run --user=root -v $(pwd):/app merimpytorch python3 save_min_max.py --data_name faulty_training --batch_size 128
