# gan
Reimplementations of GANs

- Assumes availability of [`tensorflow/models/official`](https://github.com/tensorflow/models) in `PYTHONPATH`. See instructions [here](https://github.com/tensorflow/models/tree/master/official#requirements).

## MNIST DCGAN
- File: [`dcganmnist/mnist.py`](dcganmnist/mnist.py)
- Inspired by DCGAN architecture, but altered the architecture based on what I found in [`osh/KerasGAN`](https://github.com/osh/KerasGAN).

## MNIST semi-supervised learning with manifold regularization
- File: [`dcganmnist/mnist_ssl.py`](dcganmnist/mnist_ssl.py)
- See [blog post](https://medium.com/@jos.vandewolfshaar/semi-supervised-learning-with-gans-23255865d0a4).
