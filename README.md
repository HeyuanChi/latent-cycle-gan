# latent-cycle-gan

![method](.imgs/latent-cycle-gan.jpg)

This is the code for the Computer Vision 24/25ws final project.

The core code of Latent-CycleGAN models is in ./models/latent_cycle_gan_model.py.

## Prerequisites
- Linux or macOS
- Python 3.8
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/HeyuanChi/latent-cycle-gan
cd latent-cycle-gan
```
Please set the environment according to the respective internal requirements.txt

### Latent-CycleGAN train/test
- Download a CycleGAN dataset (e.g. vangogh2photo):
```bash
bash ./datasets/download_cyclegan_dataset.sh vangogh2photo
```
- Train a model:
```bash
bash ./train.sh
```

- Test the model:
```bash
bash ./test.sh
```