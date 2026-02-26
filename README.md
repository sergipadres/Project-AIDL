# Flow Matching for Disease Progression in Medical Imaging
### Project-AIDL - Final project for the Postgraduate program Artificial Intelligence with Deep Learning 2025–2026, UPC Barcelona


## Authors
- Arnau Claramunt
- Sandra Márquez
- Sergi Padrés
- Albert Vidal

Project advisor:
- Òscar Pina 

[![GitHub followers](https://img.shields.io/github/followers/ArnauCS03?label=ArnauCS03)](https://github.com/ArnauCS03) &nbsp;&nbsp; 
[![GitHub followers](https://img.shields.io/github/followers/sanmarquez?label=sanmarquez)](https://github.com/sanmarquez) &nbsp;&nbsp;
[![GitHub followers](https://img.shields.io/github/followers/sergipadres?label=sergipadres)](https://github.com/sergipadres) &nbsp;&nbsp;
[![GitHub followers](https://img.shields.io/github/followers/trsk-ndfns?label=trsk-ndfns)](https://github.com/trsk-ndfns) &nbsp;&nbsp;
[![GitHub followers](https://img.shields.io/github/followers/oscar97pina?label=oscar97pina)](https://github.com/oscar97pina) &nbsp;&nbsp;

<br>

## Project summary

**Goal:** Learn a generative transformation that takes a healthy medical image and produces a partially-ill / progressively-ill image, doing an interpolation along a disease trajectory with Flow Matching.

**Image → encode to latent → Flow Matching in latent space → decode to images**



## Datasets 
Initial dataset to test:

[**PneumoniaMNIST**](https://medmnist.com/), it contains 5,856 pediatric chest X-Ray images. And we used 224x224 images.

<br>

## Development

We iterated on the reconstruction model in a few steps:

### 1. **ViT Autoencoder (ViT encoder + ViT decoder)**  
   Our first baseline used a Vision Transformer for both the encoder and the decoder.  
   **Result:** reconstructions were very noisy and unstable.

### 2. **Masked Autoencoder variant (MAE-style)**  
   We then introduced masking during training (masking a fraction of patches/tokens).  
   **Result:** the model learned meaningful reconstructions but blur remained.

### 3. **Switch decoder to CNN (ViT encoder + CNN decoder)**  
   To improve image-level detail and stability, we replaced the transformer decoder with a convolutional decoder (transpose-conv upsampling).  
   **Result:** reconstructions became more stable, but still **blurry**.

![Final Reconstruction](https://github.com/sergipadres/Project-AIDL/blob/main/assets/reconstruccio_final.png?raw=true)

After doing a little bit of hyperparameter tuning and improving some logic:

![Final Reconstruction](https://github.com/sergipadres/Project-AIDL/blob/main/assets/reconstruccio_embed_dim64_mask0.2_latent16_epoch100.png?raw=true)

**Sanity check:** We also ran an “overfit one image” experiment (single sample, no masking) to verify that the autoencoder and training loop can learn a near-perfect reconstruction and confirm the autoencoder works as intended.

### 4. **Deterministic U-Net Autoencoder (ResNet18 encoder) + "Skip Dropout"**
- Outcome: reconstructions structurally correct
- Problem: deterministic latent was **discontinuous** (“holes between patients”)
- Conclusion: **not suitable** as a generative latent space (sampling/interpolation off-manifold)

### 5. **Spatial VAE (structured latent tensor 4×7×7)**
- Latent is a **feature map** (spatially organized), not a flat vector
- Probabilistic latent + reparameterization → smoother / more continuous manifold
- Strong candidate for Flow Matching because it is more likely to be continuous + sampleable

### 6. **Latent interpolation continuity test (implemented + plotted)**
- Pick one Healthy and one Pneumonia sample from validation
- Encode both and keep only μ (mean latent)
- Linearly interpolate between the two μ’s in a few steps
- Decode each step with skip features zeroed (use encoder only to get feature shapes, then remove skip information and inject the interpolated latent at the deepest level)
- Plot the interpolated decoded images
- Print the latent distance between the two endpoints

The decoded images evolve smoothly along the latent interpolation from healthy to pneumonia, suggesting the Spatial VAE latent space is reasonably continuous. This makes the latent space a suitable candidate for Flow Matching.

### 7. **Save the latents**

To make experiments reproducible and avoid re-encoding every time, we uploaded **precomputed train latents + labels** to `assets/`.  
These `.npy` files can be loaded directly (NumPy → Torch tensor) and used to train Flow Matching models without needing the encoder.

We include both:
- **Flat vector latents (256-D)** from early experiments (best paired with an MLP-based flow)
- **Spatial latents** from the latest **pure Spatial VAE without skip connections** (recommended), e.g. `latents_pure_train.npy` (+ matching labels).


### 8. **FlowCNN (latent-space flow model)**
- Having the Spatial VAE that learns the latent.
- Train  a **FlowCNN** that learns a **velocity field** in latent space. "How to move latent z at time t"
- This is a global (unpaired) Flow Matching model: it learns a transformation from the healthy latent distribution toward the pneumonia latent distribution in general, not a patient-specific or paired (healthy→the same patient’s pneumonia) progression.
- At inference: pick one healthy latent, integrate ODE with Euler, decode each intermediate latent back into images

The figure shows a generated **latent-space trajectory**: starting from a **healthy μ-latent**, we integrate the learned Flow Matching **velocity field** over time and decode the intermediate latents back into images, we transition toward **pneumonia-like** reconstructions:

![Flow trajectory](https://github.com/sergipadres/Project-AIDL/blob/main/assets/trajectoria_FM_v1.png?raw=true)


### 9. **Metric Flow Matching**

We add a manifold-awareness layer on top of baseline Flow Matching to avoid off-manifold latent trajectories.

- **Latent extraction:** encode the dataset into latent vectors (μ latents from our autoencoder/VAE).
- **Metric learning (RBFMetric):** fit **k-means** on the latent dataset and build an **RBF-based proximity score** that outputs high values for latents close to the data distribution and low values for out-of-distribution latents.
- **Geodesic correction (Gamma):** train a small network that **bends the straight-line interpolant** between (healthy → pneumonia) latents so intermediate points stay in regions where the metric is high (i.e., closer to the latent manifold).
- **Vector field training:** compute training targets (conditional flows) along these **metric-corrected interpolants** and train the final **flow model** to predict the latent velocity field.

**Key idea:** instead of learning trajectories that may cut through unrealistic latent regions, we learn trajectories that remain close to the latent distribution, improving realism when decoding intermediate states.


### 10. **Multi-stage Autoencoder (latent = 98)**

Added a **multi-stage autoencoder** that reconstructs from a **98-dim latent** using **interpolated upscaling**.  
Interpolation-based upsampling allows feature maps to grow in *fractional* steps.

Key additions:
- **Interpolated upscaling** (multi-stage) for smoother detail recovery.
- **CBAM attention (Convolutional Block Attention Module)** to focus reconstruction on the most relevant feature-map regions
- **Masking update:** previously we used a fixed mask ratio (0.5). Now the mask is a **learned parameter**.
- **RoPE embedding update** in the encoder.
- **Loss:** trained with **L1 + Perceptual Similarity (LPIPS)**

### 11. **Spatial VAE High-Res (v2.0) — Pure VAE, Zero Skips**

**New architecture & features**
- **Pure VAE (Zero Skips):** full removal of U-Net skip connections. The model becomes a real bottleneck that forces semantic compression into the latent space.
- **Higher-res latent:** we modified the ResNet18 feature extraction by cutting at **Layer 3** (instead of the final stage), increasing the latent map from **4 × 7 × 7 → 4 × 14 × 14** to preserve pathology shape detail.
- **Final Sigmoid layer:** decoder output is constrained to **[0, 1]**, preventing contrast artifacts (gray-ish pixels) in reconstructions/interpolations.
- **VGG16 perceptual loss** + **L1** (with ImageNet normalization), replacing classic MSE. This penalizes texture loss and reduces the typical VAE blur, yielding sharper ribs/edges.
- **Memory management:** implemented **Gradient Accumulation** + **Automatic Mixed Precision (torch.cuda.amp)** to simulate larger batch sizes (e.g., 32) on ~14GB GPUs for 224×224 training.

**Validation & data extraction**
- **Spatial sanity check (“Frankenstein Test”):** passed. We fused half of a “Healthy” latent with half of a “Pneumonia” latent and decoded a half-healthy/half-ill image, showing the latent preserves 2D topology.

### 12. **VAE High-Res Upgrade + TorchXRayVision Integration (v2)**

Custom VAE High-Res now with latents of (4 × 28 × 28). We optimized the base ResNet18 VAE to maximize fine anatomical structure retention, trading compression for visual fidelity.

Visual impact: sharper anatomy (bone structures + cardiac borders) with fewer blur artifacts.

Latent PCA: baseline class separation with centroid Euclidean distance 26.51.

**Clinical Expert VAE (Transfer Learning with TorchXRayVision)**

We replaced the generic encoder with a medical-domain expert model to capture *clinically relevant* patterns (pathology) instead of low-level textures.

- **Model surgery:** integrated `XRV-ResNetAE-101-elastic` (torchxrayvision), extracting **512-channel** tensors while keeping **28 × 28** resolution.
- **Two-phase training:**
  - 20 epochs warm-up (frozen encoder)
  - 60 epochs **fine-tuning** (full unfreeze), LR = `1e-5`
  - **Early stopping** converged around epoch ~80 (initial plan was 100 epochs)
  - **PCA centroid distance:** **30.97** (substantial increase in pathology separability)

#### Pipeline improvements
- **Automatic Early Stopping:** patience = 15 epochs (reduced overfitting + compute time).


### 13. Metric Flow Matching (MFM) v2 — Unpaired Patients + Flat Latents

Goal: train the final **vector field** (latent velocity) for Metric Flow Matching. After this code, everything is ready to use the vector field and generate intermediate latents using the flat_latent_ode_inference_unpaired code.

What changed in v2:
- **Unpaired setting:** we do **not** use endpoint-conditioned (patient-specific) pairs. We learn a global disease-direction flow.
- **Flat latent vectors:** experiments start with the **multi-stage AE** flat latent (98 dims), before extending to other latent types. And add the class definition of this autoencoder and load the state_dict weights to generate the latents on this code.


<br>

> [!WARNING]  
> Disclaimer: This repository is research/educational work. Generated images are not intended for clinical use.
