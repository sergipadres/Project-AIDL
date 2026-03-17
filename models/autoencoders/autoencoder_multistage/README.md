## Vision Transformer Masked Autoencoder

The autoencoder is formed by a masked Vision Transformers Encoder, that operates on convolutionally embedded patches of images and has configurable number of attention and MLP layers, embedding dimension, latent size dimension and masking percentage.  
The decoder brings back the image by transforming the latent representation into 7x7 feature maps that are progressively up-scaled and merged by alternating fractional upscaling layers and convolutions accompanied by Channel Based Attention Modules.  
The autoencoder is trained on a reconstruction loss and a VGG perceptual loss.