#!/bin/bash
#SBATCH --job-name distr_TF
#SBATCH -D .
#SBATCH --output distr_B0_1GPU_.out
#SBATCH --error distr_B0_1GPU_.err
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --gres=gpu:1
#SBATCH --time 20:00:00

module purge; module load gcc/8.3.0 cuda/10.1 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 fftw/3.3.8 ffmpeg/4.2.1 opencv/4.1.1 atlas/3.10.3 scalapack/2.0.2 szip/2.1.1 python/3.7.4_ML

export PYTHONUNBUFFERED=1

python distr_effnet_shear.py --image_model 0 --optimizer sgd --epochs 20
