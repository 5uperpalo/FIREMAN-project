#!/bin/sh
#SBATCH -n 10            # 8 cores
#SBATCH -t 3-03:00:00   # 1 day and 3 hours
#SBATCH -p compute      # partition name
#SBATCH -J my_job_name  # sensible name for the job

#PROOT_NO_SECCOMP=1 $HOME/udocker/udocker run -v $HOME/help/GAIN/:/srv merimkeras python main_letter_spam.py --data_name tennesee_dat
source /etc/profile.d/modules.sh
module load staskfarm

staskfarm ./bash_scripts/commands_train.txt
#PROOT_NO_SECCOMP=1 $HOME/udocker/udocker run -v $(pwd):/srv merimkeras python prepare_fault_free_training.py --data_name faulty_testing --batch_size 128
