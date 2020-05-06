# Running distributed training of the image model (CTE-POWER machine)

- Download https://github.com/javiferran/document-classification repository and move it to CTE:

```bash
$ cd Downloads
$ git clone https://github.com/javiferran/document-classification.git
$ scp Downloads/document-classification bsc31275@plogin1.bsc.es:/gpfs/home/bsc31/bsc31275
```

## TensorFlow
- Download efficientnet.tfkeras models:

```bash
https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5
```

Remember we need the ones with _notop_ sufix.

- Create directory to store models:

```bash
$ ssh bsc31275@plogin1.bsc.es "mkdir -p /gpfs/home/bsc31/bsc3175/document-classification/image_model/tensorflow/.keras/models"
```

- Copy model to created directory:

```bash
$ scp Downloads/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5 bsc31275@plogin1.bsc.es:/gpfs/home/bsc31/bsc31275//document-classification/image_model/tensorflow/.keras/models
```

- If efficientnet libary is not installed:

Download https://github.com/qubvel/efficientnet repository and copy it in bsc31275@plogin1.bsc.es:/gpfs/home/bsc31/bsc31275/document-classification/image_model/tensorflow


- Run job.sh

Select the desidered number of gpus by changing `ntasks` and `gres=gpu:` values. 

### TensorFlow with Horovod

- Run job_hvd.sh

## PyTorch

- Download efficientnet-pytorch models:

```bash
'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
```

- Create directory to store models:

```bash
$ ssh bsc31275@plogin1.bsc.es "mkdir -p /gpfs/home/bsc31/bsc31275/document-classification/image_model/.cache/torch/checkpoints"
```

- Copy model To created directory:

```bash
$ scp Downloads/efficientnet-b0-355c32eb.pth bsc31275@plogin1.bsc.es:/gpfs/home/bsc31/bsc31275/document-classification/image_model/.cache/torch/checkpoints
```

- Run JOB_parallel.sh

Select the desidered number of gpus by changing `ntasks` and `gres gpu:` values. 

