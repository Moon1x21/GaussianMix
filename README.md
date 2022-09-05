
# GaussianMix: Data Augmentation focusing the structure of the Convolution Network

## Prerequisites

* Python 3.5
* PyTorch 1.0
* GPU (recommended)

## Datasets

* CIFAR-10/100: automatically downloaded by PyTorch scripts to `data` folder
* ImageNet: manually downloaded from [ImageNet](http://www.image-net.org/) (ILSVRC2012 version) and moved to `train` and `val` folders in your `dataroot` path (e.g., `./imagenet/`)

## How to Train

Our script occupies all available GPUs. Please set environment `CUDA_VISIBLE_DEVICES`.

### CIFAR-10 and WideResNet28-10

with SICAP

```bash
python main.py --dataset cifar10 --model WideResNetDropout --depth 28 --params 10 --beta_of_sicap 1.0 --postfix sicap1.0
```

without SICAP

```bash
python main.py --dataset cifar10 --model WideResNetDropout --depth 28 --params 10
```

We trained these models on a single GPU (GeForce GTX 1080).

### CIFAR-100 and WideResNet28-10

with SICAP

```bash
python main.py --dataset cifar100 --model WideResNetDropout --depth 28 --params 10 --beta_of_sicap 0.3 --postfix SICAP1.0
```

without SICAP

```bash
python main.py --dataset cifar100 --model WideResNetDropout --depth 28 --params 10
```

We trained these models on a single GPU (GeForce GTX 1080).


### ImageNet and WideResNetBottleneck50-2 for 100 epochs

with SICAP

```bash
python main.py --dataset ImageNet --dataroot [your imagenet folder path(like ./imagenet)] --model WideResNetBottleneck --depth 50 --epoch 100 --adlr 30,60,90 --droplr 0.1 --wd 1e-4 --batch 256 --params 2 --beta_of_ricap 0.3 --postfix ricap0.3
```

without SICAP

```bash
python main.py --dataset ImageNet --dataroot [your imagenet folder path(like ./imagenet)] --model WideResNetBottleneck --depth 50 --epoch 100 --adlr 30,60,90 --droplr 0.1 --wd 1e-4 --batch 256 --params 2
```
