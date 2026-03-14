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
---

## 1. Project Summary & Problem Statement

**Goal:** Learn a generative transformation that takes a healthy medical image and produces a partially-ill / progressively-ill image, performing an interpolation along a disease trajectory using Flow Matching.

**The Pipeline:** Image → Encode to latent space → Flow Matching in latent space → Decode to pathological image.

Conditional Flow Matching (CFM) is a deep generative model framework that allows for generating samples from one distribution given samples from another, learning a mapping between both distributions parameterized by a vector field. It provides a simulation-free learning step, improving training efficiency over diffusion models. Our main hypothesis is that CFM is an optimal approach to learn trajectories from a healthy distribution to a sick distribution in the medical imaging domain, specifically utilizing [**PneumoniaMNIST**](https://medmnist.com/) (pediatric chest X-rays), it contains 5,856 pediatric chest X-Ray images. And we used 224x224 images.

We address the high computational cost and artifact generation typical of pixel-space integration by projecting the data into a continuous, regularized latent space.

---


##  2. Experiments & Key Results

This project is structured around two main phases. We evaluate our pipeline from latent space optimization to the final disease progression synthesis.

### Phase 1: Latent Space Topology
#### Experiment 2: Spatial VAE vs. Unregularized ViT
* **Hypothesis:** Retaining spatial feature maps via a regularized Convolutional VAE enforcing a $\mathcal{N}(0, I)$ prior yields a denser, more continuous latent topology compared to 1D sequence tokenization (ViT), preventing Out-of-Distribution (OOD) querying during generative sampling.
* **Setup (Dataset & Model):** PneumoniaMNIST (pediatric chest X-rays, 224x224). We compared a Custom Spatial VAE (utilizing `XRV-ResNetAE-101-elastic` as the encoder) against a Multistage Masked Vision Transformer (ViT) Autoencoder.
* **Results:** The Spatial VAE successfully enforced the Gaussian prior, revealing natural unsupervised clustering of pathology. Conversely, the ViT exhibited severe dimensional collapse. 
  * *Structural Fidelity (SSIM):* Spatial VAE achieved **0.88**, vastly outperforming the ViT (**0.62**).
* **Conclusions:** Spatial VAEs provide the optimal, unobstructed continuous path required for downstream Flow Matching, successfully preserving the anatomical proportions of the thoracic cavity.

`![PCA Latent Projections](https://github.com/sergipadres/Project-AIDL/blob/main/assets/2D_PCA_Projection_Latent_Spaces.png)`

**Sanity check:** We also ran an “overfit one image” experiment (single sample, no masking) to verify that the autoencoder and training loop can learn a near-perfect reconstruction and confirm the autoencoder works as intended.

---

### Phase 2: Flow Matching Integration
#### Experiment 1: Baseline Conditional Flow Matching (Pixel Space)
* **Hypothesis:** Conditional Flow Matching (CFM) can successfully learn a vector field that maps a healthy X-ray distribution to a pneumonia-infected distribution.
* **Setup (Dataset & Model):** CFM integration applied directly on the pixel space of the PneumoniaMNIST dataset using a standard neural ODE formulation.
* **Results:** While the model learned the general direction of the disease progression, pixel-level integration proved to be computationally heavy. The trajectories often introduced blurring and structural artifacts due to the high dimensionality of the image space.
* **Conclusions:** Pixel-space Flow Matching is inefficient and prone to generating anatomically inconsistent images. Dimensionality reduction is strictly required.

#### Experiment 3: Latent Space Flow Matching (The Proposed Pipeline)
* **Hypothesis:** Executing the Flow Matching ODE/RK4 solver over the optimally regularized latents of the Spatial VAE will eliminate OOD artifacts and generate highly realistic, continuous interpolations of disease progression.
* **Setup (Dataset & Model):** A Vector Field MLP/FlowCNN was trained on the extracted latents of both the Spatial VAE and the ViT (for ablation). During inference, we used an Euler ODE solver to push healthy latents towards the pathological distribution, decoding the final steps back to image space.
* **Results:** * *ViT Latent Flow:* The vector field struggled with the collapsed grid-like topology, resulting in noisy, inconsistent trajectories.
  * *Spatial VAE Latent Flow:* The model successfully navigated the continuous manifold. The interpolations smoothly injected pathological features (opacity) without destroying the structural integrity of the ribs and lungs.
* **Conclusions:** Latent-space Flow Matching paired with spatial regularization (Spatial VAE) is a vastly superior framework for modeling disease progression in medical imaging, combining computational efficiency with high generative fidelity.

 `![Flow Matching Trajectories](https://github.com/sergipadres/Project-AIDL/blob/main/assets/experiment3_trajectories_images.png)`

*(Note: For a detailed log of intermediate versions, failed approaches, and early testing—such as our ViT MAE tests—please refer to the `CHANGELOG.md` file and the `notebooks/development/` folder).*

---

## 3. Repository Structure

```text
pneumonia-flow-matching/
├── assets/                     # Final data and visual resources
│   └── images/                 # PCA plots and reconstructions for the README
├── checkpoints/                # EMPTY FOLDER (Place downloaded .pth weights here)
├── models/                     # Python scripts (Architectures and Functions)
│   ├── autoencoders/           
│   └── flow_matching/          
├── notebooks/                  # Jupyter Notebooks
│   ├── development/            # Previous versions, deprecated tests, and sanity checks
│   ├── training/               # Training loops for VAE and Flow Matching
│   └── experiments/            # Final pipeline experiments (Exp 1, 2, 3 and 4)     
├── CHANGELOG.md                # Log of versions, trials, and model surgery
├── requirements.txt            # Python dependencies
└── README.md                   # Summary and instructions

```
---

## 4. Installation

* Clone this repository:

git clone [https://github.com/](https://github.com/)[USER_NAME]/[REPO-NAME].git
cd [REPO-NAME]

* Install dependencies:

pip install -r requirements.txt

---

## 5. Model Weights & Local Setup (Important)
Due to GitHub's file size limits, the pre-trained weights (.pth files) for the Spatial VAE and Flow Matching models are hosted externally. To run the evaluation notebooks without training from scratch:

Download the pre-trained weights from this Google Drive link: [INSERT_DRIVE_LINK_HERE].

Place the downloaded .pth files inside the checkpoints/ folder of this repository.

--- 
## 6. How to Run

Phase 1 (Latent Space Validation): Open notebooks/evaluation/experiment-2-multistage-vs-spatialvae.ipynb to reproduce the latent space topology analysis (PCA) and reconstruction metrics.

Phase 2 (Flow Matching Interpolation): Open notebooks/evaluation/experiment3.ipynb to visualize the continuous translation from healthy to pneumonia using the ODE solver in latent space.
