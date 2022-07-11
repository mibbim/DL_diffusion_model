# DL_diffusion_model flow

## Dataset

- [x] MNIST
- [ ] Cifar10
- [ ] Vettori numerici

## Trainer

- [x] Trainstep method
- [x] Train method
- [x] Performance Meter
- [ ] Model Save/Load
- [ ] Check Point / Tensorboard
- [ ] Evaluation method / Evaluator Object

## DiffusionModel

### UNet o stato dell'arte CNNs x image segmentation

Examples:

    1. Small: https://github.com/dbasso98/GANs/blob/main/CycleGANs/generator.py
    2. Large: https://colab.research.google.com/drive/1NFxjNI-UIR7Ku0KERmv7Yb_586vHQW43?usp=sharing#scrollTo=fc4IdL5YawTN

### Loss

- [ ] MSE noise: $L_{simple}$
- [ ] Add $L_{VLB}$: $L_{simple}+\lambda L_{VLB}$
- [ ] Cosine Schedule
- [ ] Add covariance matrix
- [ ] Improve gradient noise

### Ref

- https://arxiv.org/pdf/2102.09672.pdf
- https://arxiv.org/pdf/2105.05233.pdf
- https://colab.research.google.com/drive/1NFxjNI-UIR7Ku0KERmv7Yb_586vHQW43?usp=sharing#scrollTo=w553nqFVg8uk

## Possible Future Implementations

- DDIM
