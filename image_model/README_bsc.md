# Running parallel training of the image model in Pytorch (CTE-POWER machine)

## Preparation

Since _efficientnet_pytorch_ library downloads the models in .cache/torch/checkpoints, and CTE_POWER has no internet connection, we have to add the models manually. First, we create the directory to store efficientnet models

```bash
ssh bsc31991@plogin1.bsc.es "mkdir -p /gpfs/home/bsc31/bsc31991/image_model_pytorch/.cache/torch/checkpoints"
```

Then we download efficientnet pretrained models in a local computer from the following links:

```bash
'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth',
'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth',
'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth',
'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth',
'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth'
```

Finally, we copy the downloaded models to the created folder (assuming it downloads in 'Downloads' folder)

```bash
scp Downloads/efficientnet-b1-f1951068.pth bsc31991@plogin1.bsc.es:/gpfs/home/bsc31/bsc31991/image_model_pytorch/.cache/torch/checkpoints/
```

Now we can run the JOB_parallel.sh file.
The important things to know before submitting the job are the following ones:

```bash
#SBATCH --output /gpfs/home/bsc31/bsc31275/logs/%j.out <- folder to store the logs .out (select user and create folder if necessary)

#SBATCH --error /gpfs/home/bsc31/bsc31275/logs/%j.err <- folder to store the logs .err (select user and create folder if necessary)

#SBATCH --gres gpu:1 <- number of gpus to be used (1, 2, 3 or 4)

#SBATCH -c 40 <- number of cpus, 40*number of gpus (40, 80, 120 or 160)

#SBATCH --time 00:10:00 <- maximum time to run the job (48:00:00 max)

python eff_big_training.py \
	--epochs 20 \ <- number of epochs
	--eff_model b0 \ <- model to use
	--load_path /gpfs/scratch/bsc31/bsc31275/ <- path to load the dataset (don't change)
```
