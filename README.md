# Flow Matching for Disease Progression in Medical Imaging
### Project-AIDL - Final project for the Postgraduate program Artificial Intelligence with Deep Learning 2025–2026, UPC Barcelona


## Authors
- Arnau Claramunt
- Sandra Márquez
- Sergi Padrés
- Albert Vi

Project advisor:
- Òscar Pina 

[![GitHub followers](https://img.shields.io/github/followers/ArnauCS03?label=ArnauCS03)](https://github.com/ArnauCS03) &nbsp;&nbsp; 
[![GitHub followers](https://img.shields.io/github/followers/sanmarquez?label=sanmarquez)](https://github.com/sanmarquez) &nbsp;&nbsp;
[![GitHub followers](https://img.shields.io/github/followers/sergipadres?label=sergipadres)](https://github.com/sergipadres) &nbsp;&nbsp;
[![GitHub followers](https://img.shields.io/github/followers/trsk-ndfns?label=trsk-ndfns)](https://github.com/trsk-ndfns) &nbsp;&nbsp;
[![GitHub followers](https://img.shields.io/github/followers/oscar97pina?label=oscar97pina)](https://github.com/oscar97pina) &nbsp;&nbsp;

<br>

## Project summary

Learn a generative transformation that takes a healthy medical image and produces a partially-ill / progressively-ill image, doing an interpolation along a disease trajectory with Flow Matching.


## Datasets 
Initial dataset to test:

[**PneumoniaMNIST**](https://medmnist.com/), it contains 5,856 pediatric chest X-Ray images. And we used 224x224 images.

<br>

## Initial development

We iterated on the reconstruction model in a few steps:

1. **ViT Autoencoder (ViT encoder + ViT decoder)**  
   Our first baseline used a Vision Transformer for both the encoder and the decoder.  
   **Result:** reconstructions were very noisy and unstable.

2. **Masked Autoencoder variant (MAE-style)**  
   We then introduced masking during training (masking a fraction of patches/tokens).  
   **Result:** the model learned meaningful reconstructions but blur remained.

3. **Switch decoder to CNN (ViT encoder + CNN decoder)**  
   To improve image-level detail and stability, we replaced the transformer decoder with a convolutional decoder (transpose-conv upsampling).  
   **Result:** reconstructions became more stable, but still **blurry**.

![Final Reconstruction](https://github.com/sergipadres/Project-AIDL/blob/main/assets/reconstruccio_final.png?raw=true)

After doing a little bit of hyperparameter tunning and improving some logic:

![Final Reconstruction](https://github.com/sergipadres/Project-AIDL/blob/main/assets/reconstruccio_embed_dim64_mask0.2_latent16_epoch100.png?raw=true)

**Sanity check:** We also ran an “overfit one image” experiment (single sample, no masking) to verify that the autoencoder and training loop can learn a near-perfect reconstruction and confirm the autoencoder works as intended.

<br>

> [!WARNING]  
> Disclaimer: This repository is research/educational work. Generated images are not intended for clinical use.
