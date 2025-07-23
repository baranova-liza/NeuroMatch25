#!/bin/bash
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:59:00
#SBATCH --mem-per-cpu=16000M
#SBATCH --account=cosc016682
#SBATCH --array=1
#SBATCH --job-name=ForceField_test_ff_%a
#SBATCH --output=ForceField_test_ff%a.out

cd ${SLURM_SUBMIT_DIR}

source /user/home/as15635/NeuroMatch/NeuroMatch25/.venv/bin/activate

echo JOB ID: ${SLURM_JOBID}

echo SLURM ARRAY ID: ${SLURM_ARRAY_TASK_ID}

echo Working Directory: $(pwd)

echo Start Time: $(date)


python -u Run_ForceField_RandomReaching.py --task_id ${SLURM_ARRAY_TASK_ID} --num_tasks 1 --num_epochs 500 --num_samples 1000 --batch_size 64 --learning_rate 0.001 --weight_decay 0.00000001 --latent_size 128 --force_field_strength_x "-1.0" --force_field_strength_y "0.0" --force_field_bool TRUE


echo Finish Time: $(date)

