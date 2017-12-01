PyTorch: Gated PixelCNN Conditional Autoregressive Decoder
==========================================================

Implementation of ["Conditional Image Generation with PixelCNN Decoders" by van den Oord et al. 2016](https://arxiv.org/abs/1606.05328)
with some modifications suggested by ["Generating Interpretable Images with Controllable Structure" by Reed et al. 2017](http://www.scottreed.info/files/iclr2017.pdf).

NOTE: currently, this project does not support masking and autoregression across multiple channels. As such, it can only model
the generation of 1 channel 2D tensors. I may implement this in the future and contributions are welcome.

The implementation is heavily based on the [Chainer implementation by Sergei Turukin](http://sergeiturukin.com/2017/02/24/gated-pixelcnn.html) and his helpful [blog post](http://sergeiturukin.com/2017/02/24/gated-pixelcnn.html).

Example usage and samples from a trained model will be added eventually; I'm in the process of using this for a class final project and need to focus on that for the time being but will come back and flesh out this repo when I have time in a few weeks.

## Important note on masks

I use slightly different convolution masks than van den Oord and Turukin. Please take a look at 
[this notebook](./note-note-on-conv-masking.ipynb) to see my explanation of the masks and how they
prevent blindspots.
