#!/bin/bash


#SBATCH --job-name b0-1
#SBATCH -D /gpfs/home/bsc31/bsc31275
#SBATCH --output /gpfs/home/bsc31/bsc31275/logs/%j.out
#SBATCH --error /gpfs/home/bsc31/bsc31275/logs/%j.err
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --ntasks 1
#SBATCH -c 40
#SBATCH --time 00:10:00

module purge;module load gcc/8.3.0 cuda/10.1 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 fftw/3.3.8 ffmpeg/4.2.1 opencv/4.1.1 atlas/3.10.3 scalapack/2.0.2 szip/2.1.1 python/3.7.4_ML

export PYTHONUNBUFFERED=1
export SLURM_MPI_TYPE=openmpi

python eff_big_training.py \
	--epochs 20 \
	--eff_model b0 \
	--load_path /gpfs/scratch/bsc31/bsc31275/
