#!/bin/bash
#SBATCH --job-name hvd_tf2
#SBATCH -D .
#SBATCH --output hvd_tf2_%j.out
#SBATCH --error hvd_tf2_%j.err
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --gres=gpu:1
#SBATCH --time 00:50:00
##SBATCH --qos=debug

module purge; module load gcc/8.3.0 cuda/10.1 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 fftw/3.3.8 ffmpeg/4.2.1 opencv/4.1.1 atlas/3.10.3 scalapack/2.0.2 szip/2.1.1 pytho
n/3.7.4_ML
#module purge; module load gcc/6.4.0 cuda/9.1 cudnn/7.1.3 openmpi/3.0.0 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.7 szip/2.1.1 ffmpeg/4.0.2 opencv/3.4.1 python/3.6.5_ML

export PYTHONUNBUFFERED=1
date
export SLURM_MPI_TYPE=openmpi
mpirun -np 1 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python horovod_effnet.py --epochs 20 --optimizer sgd --image_model 0
