# DL_diffusion_model flow

## Dataset

- [x] MNIST
- [ ] Fashions mnist?
- [ ] celebA?
- [ ] Cifar10
- [ ] Vettori numerici

## Trainer

- [x] Trainstep method
- [x] Train method
- [x] Test train method
- [x] Performance Meter
- [X] Model Save / Load
- [x] Check Point / Tensorboard
- [ ] Evaluation method / Evaluator Object (maybe not necessary)

## DiffusionModel

### UNet o stato dell'arte CNNs x image segmentation

- [X] Implement Unet
    - [X] Upsampling/ Downsampling bulding blocks
    - [X] Choose Unet architecture aka number of layers
    - [X] Use different activation functions
    - [X] Fix maxpooling for odd input size of images adding a padding
    - [X] Add dropout -> ask for where?
    - [ ] Change weights initialization with Xavier and He et. alt.
- [ ] Time Embeddings  (from https://colab.research.google.com/drive/1NFxjNI-UIR7Ku0KERmv7Yb_586vHQW43?usp=sharing#scrollTo=aHwkcmvkRLH0 https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py and https://huggingface.co/blog/annotated-diffusion implementations)
    - [X] Sinusoidal Encoder 
    - [X] MPL for time embedding in DoubleConv class
    - [X] Attention block
    - [ ] Usage of Attention block in ConvBlockDownsample and ConvBlockUpsample

Examples:

    1. Small(no time embedding): https://github.com/dbasso98/GANs/blob/main/CycleGANs/generator.py
    2. Large(with time embedding): https://colab.research.google.com/drive/1NFxjNI-UIR7Ku0KERmv7Yb_586vHQW43?usp=sharing#scrollTo=fc4IdL5YawTN
    3. Simple(especially time embedding): https://huggingface.co/blog/annotated-diffusion

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
- https://arxiv.org/pdf/1706.03762.pdf

## Possible Future Implementations

- DDIM
