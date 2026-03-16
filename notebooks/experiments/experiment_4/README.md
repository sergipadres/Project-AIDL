# Experiment 4: Metric-Regularized Flow Matching

## Objective
The goal of this experiment is to model continuous disease progression in chest X-ray images from the **PneumoniaMNIST** dataset. We aim to learn a vector field in latent space that transforms healthy images into pathological ones, ensuring the trajectories remain anatomically consistent and stay within the data manifold.

## Hypothesis
Incorporating an **explicit learned latent metric** together with a **gamma-corrected interpolation path** will:
1. Stabilize Flow Matching training.
2. Prevent trajectories from leaving the data manifold.
3. Produce smooth, anatomically consistent disease progressions with minimal artifacts when combined with a spatial vector field (U-Net).

---

## Training Workflow
To achieve a stable flow, the training process is decoupled into sequential stages:
1. **Metric Training:** First, we learn a latent-space metric that assigns lower penalty to points closer to the learned data manifold.
2. **Gamma Calibration:** Using the learned metric, we compute the gamma-corrected interpolation path.
3. **Vector Field Training:** Finally, we train the vector field $v(z, t)$ using both the metric and the gamma-corrected path.
4. **Inference:** We use the trained Autoencoder to map images to latents, apply the metric-regularized vector field to guide transitions, and solve the ODE via **RK4** to visualize anatomically consistent intermediate steps.

---

## Model Weights & Requirements
To execute inference and trajectory generation, ensure the following checkpoints are available (hosted on Drive):

* **Autoencoder**: `VIT-BCE-VGG-256-epoch078`
* **Vector Field**: `vector_field.pt`
* **Validation**: `Dino-classifier-epoch008.pt` (Binary classifier for pneumonia detection)

---

## Latent Representations
The study compares two distinct latent architectures:

### 1. Flat Latent Space
* **Shape:** $z \in \mathbb{R}^{256}$
* **Encoder:** ViT-based masked autoencoder (MAE).
* **Vector Field:** MLP-based.
* **Characteristics:** Low dimensional and stable, but lacks explicit spatial structure.

### 2. Spatial Latent Space
* **Shape:** $z \in \mathbb{R}^{4 \times 28 \times 28}$
* **Purpose:** Preserves spatial structure for localized pathology.
* **Vector Field:** U-Net-based operating on the spatial grid.
* **Characteristics:** More expressive dynamics for localized disease features.

---

## Flow Matching & Integration
The vector field $v(z, t)$ approximates the velocity of the interpolation path:
$$\frac{dz}{dt} = v(z, t)$$
Regularizing the latent metric ensures that latent movement corresponds meaningfully to morphological changes. At inference, we solve this using **Runge-Kutta 4 (RK4)** to ensure stability as we transition:
> **Healthy** $\rightarrow$ **Anatomically Consistent Transition** $\rightarrow$ **Pneumonia**

---

## Evaluation with DINO Classifier
We use a frozen **DINOv2-based classifier** to audit the progression. It predicts the probability of pneumonia $P(\text{pneumonia})$ for decoded intermediate images. 

* **Validation Goal:** A smooth, monotonic increase in probability from $t=0$ to $t=1$.
* **Metric Comparison:** Comparing "flat" vs. "spatial" flows to identify which maintains better manifold consistency and avoids non-medical artifacts.