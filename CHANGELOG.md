# Optimització Major: Arquitectura Híbrida i Càrrega VRAM

Aquesta versió introdueix canvis estructurals crítics per solucionar colls d'ampolla de rendiment i problemes d'estabilitat numèrica (NaNs), a més de millorar la qualitat visual de les reconstruccions mèdiques.

## Resum de Rendiment
El model ha convergit satisfactòriament després de 100 èpoques, mostrant una reducció dràstica de l'error sense signes d'overfitting (la *Val Loss* es manté enganxada a la *Train Loss*).

| Mètrica | Inici (Epoch 1) | Final (Epoch 100) | Millora |
| :--- | :--- | :--- | :--- |
| **Train Loss** | 0.06383 | **0.00349** | -94.5% |
| **Val Loss** | 0.04433 | **0.00406** |  -90.8% |

###  Resultats Visuals
Comparativa entre la radiografia original i la reconstrucció del nou Decoder CNN:
![Reconstrucció Final](assets/reconstruccio_final.png)
*(Nota: S'observa la millora en textures orgàniques i absència d'artifacts de tipus quadrícula)*

---

## Llista Detallada de Canvis

### 1. Gestió de Dades (Data Pipeline) 
* **Implementació de `TurboDataset` (Càrrega a VRAM):**
    * **Canvi:** S'ha eliminat la càrrega dinàmica disc-CPU-GPU. Ara tot el dataset es carrega a la VRAM a l'inici.
    * **Justificació:** El dataset PneumoniaMNIST (224x224, uint8) ocupa només **~280 MB** desplegat. Amb una GPU T4 (16 GB), utilitzem menys del 2% de memòria. Això elimina totalment el coll d'ampolla i accelera l'entrenament d'hores a minuts.
* **Estabilitat `Float32`:**
    * **Canvi:** Entrada convertida a `.float()` (32-bit) en lloc de forçar `.half()` (16-bit).
    * **Justificació:** Forçar 16-bit a l'entrada causava `NaNs` i valors infinits en losses petites. L'estabilitat de 32-bit és prioritària; la velocitat es gestiona via Autocast.

### 2. Arquitectura del Model (ViT Híbrid) 
* **Nou Decoder CNN:**
    * **Canvi:** Substitució del Decoder basat en Transformers per un basat en Convolucions (`ConvTranspose2d`).
    * **Justificació:** El ViT pur patia de manca de biaix inductiu espacial, generant patrons quadrats. La CNN reconstrueix millor les textures suaus dels teixits biològics.
* **Optimització de Profunditat (Depth=6):**
    * **Canvi:** Reducció de l'Encoder de 10 a 6 capes.
    * **Justificació:** Per a un problema binari amb imatges en escala de grisos, 10 capes provocaven *overfitting* per excés de capacitat. 6 capes ofereixen l'equilibri òptim per aprendre patrons estructurals sense memoritzar soroll.
* **Masking Vectoritzat i Corregit:**
    * **Canvi:** Eliminació del bucle `for` per imatge i correcció de l'aplicació de la màscara al tensor d'entrada.
    * **Justificació:** La vectorització aprofita el paral·lelisme de la GPU. La correcció del *bug* lògic assegura que el model realment vegi imatges incompletes, forçant l'aprenentatge semàntic.

### 3. Bucle d'Entrenament (Optimization Loop) 
* **Precisió Mixta (`Autocast` + `Scaler`):**
    * **Canvi:** Ús de `torch.amp.autocast` i `GradScaler` estàndard.
    * **Justificació:** Permet operacions ràpides (convolucions en FP16) i precises (sumes/loss en FP32) automàticament, eliminant la necessitat de saltar batches amb errors.
* **Batch Size = 64:**
    * **Canvi:** Ajust de 256 a 64.
    * **Justificació:** Tot i que 256 és més ràpid, 64 proporciona un millor gradient estocàstic (soroll beneficiós) per escapar de mínims locals i generalitzar millor en test.

<br>

---

### 02/02/26

A l'inici del fitxer de codi afegiria un comentari definint una versió, perquè ens sigui més comode referenciar-la entre nosaltres. L'actual que he modificat la he anomenat v5. També posar l'arquitectura i la data. Exemple: *v5 - Masked Autoencoder ViT + CNN decoder - 02/02/2026*

### Sanity check - Overfit una imatge

Pujo el codi que he usat amb el nom: `vit_train_autoencoder_overfitting1image.ipynb` a la nova carpeta /extra.
Usa com a base el codi que va pujar la Sandra el 24 de gener.

- No he usat el masking, així mirem la reconstrucció, no inpainting. `mask_ratio=0.0`
- `weight_decay=0` perquè volem memorization. Si s'usa pot ajudar a reduir l'overfitting en datasets petits.
- No usa `OneCycleLR` així el Learning Rate és més senzill.
- No usa `scaler`
- Pel dataset `BATCH_SIZE=32` i construeix un nou dataset on només usa sola imatge repetida.

Resultat:

![Overfitting](assets/overfitting_one_image.png)

Tot i que reconstrueix bé, he fet alguns petits canvis al codi original (versió 4 que va pujar la Sandra) i ha fet que els resultats siguin menys borrosos. 


### Modificacions fitxer vit_train_autoencoder_provaSM.py:

Al `MHSelfAttentionBlock()`:

Canviar: `LayerNorm((channels, self.embed_dim))` a `nn.LayerNorm(self.embed_dim)`. Abans no es feia de la manera usual que s'usa als transformers. 

També el `self.mlp` a la última capa no fer l'activació `nn.SiLU()`. Pot reduir la qualitat al reconstruir i distorsionar el *residual stream*.

Al `forward()` modificar la lògica de l'attention, corregint les dimensions perque usi la N (tokens) abans usava heads. També afegir una variable latent_in per sumar més tard com a attention residual. I afegir que es guardi un resultat a la variable: `self.oW` que no s'usava.

<br>

A la funció `ViTEncoder()`:

Afegir com a parametre el `latent_per_patch=16` així no està hardcoded (i amb unes proves que he fet, el valor va millor usar 16 i no 2).

Canviar aquesta línia perque usi el parametre que se li passa: `size = int(N * self.mask_ratio)`

Al `forward()` el valor que posem a la mascara posar 0.0 en comptes de -1, ajuda més i és un valor dins el rang (la imatge té [0, 1]). Ara: `x[batch_indices, mask_indices, :] = 0.0`.

També sumar el `self.pos_embed` després de fer el masking, per no perdre el posicionament, així el model encara sap on pertanyia el patch que falta degut a la màscara.
A la part de `for block in self.blocks` ja no es suma la x, ja que ara el `MHSelfAttentionBlock`ja inclou internament les connexions residuals.

<br>

Al `CNNDecoder()`:

Afegir el paràmetre `patch_size=16` per no tenir-lo hardcoded.

Al `ViTMaskedAutoencoderCNN()`:

Afegir el paràmetre `latent_per_patch` a tots els llocs que tocava.

A la part de la inicialització del model afegir els nous parametres i els que m'han donat resultats menys borrosos: `model = ViTMaskedAutoencoderCNN(img_size=img_size, mask_ratio=0.2, embed_dim=64, latent_per_patch=16).to(DEVICE)`

<br>

Després dels canvis la reconstrucció queda tal que així:

![Final Reconstruction](assets/reconstruccio_embed_dim64_mask0.2_latent16_epoch100.png)

Usant els parametres: `embed_dim=64` `mask_ratio=0.2` `latent_per_patch=16` `epoch=100`


La imatge resultant la guardo a: `assets/reconstruccio_embed_dim64_mask0.2_latent16_epoch100`

### 07/02/26

### "Evolució de l'Arquitectura (Model Log)"

## Objectiu
Dissenyar un Autoencoder capaç de comprimir radiografies de tòrax (**PneumoniaMNIST**) en un espai latent compacte i continu, apte per a la generació posterior mitjançant *Flow Matching*.

---

### Versió 1: U-Net Autoencoder Determinista (Baseline)

* **Data:** 04/02/2026
* **Arquitectura:** U-Net amb backbone ResNet18.
* **Estratègia Latent:** Vector lineal $z \in \mathbb{R}^{256}$ (Compressió total).
* **Innovació:** Implementació de **"Skip Dropout"**.
    * *Problema detectat:* En arquitectures U-Net estàndard, les connexions residuals permeten filtrar informació d'alta freqüència directament al decoder, evitant que l'encoder aprengui una representació latent significativa (el model "fa trampes").
    * *Solució:* Es va aplicar un `dropout=1.0` a les connexions residuals durant la primera fase d'entrenament (bloqueig estructural) i `dropout=0.5` durant el refinament.
* **Resultat:** Reconstruccions correctes estructuralment.
* **Limitació:** L'espai latent resultant, al ser determinista, presentava discontinuïtats ("forats" entre pacients), fent-lo inviable per a la generació de noves mostres amb models de difusió.

---

###  Versió 2: Spatial VAE (Arquitectura Final - State of the Art)

* **Data:** 07/02/2026
* **Canvi de Paradigma:** Transició de vector pla a **Latent Espacial**.
* **Arquitectura:**
    * Es substitueix la capa lineal del coll d'ampolla per convolucions $1 \times 1$.
    * **Nou espai latent:** Tensor de dimensions $(4 \times 7 \times 7)$. Es manté la coherència espacial (dalt/baix, esquerra/dreta) en la representació comprimida.
    * Introducció del mostreig probabilístic (*Reparameterization Trick*) per garantir continuïtat (manifold suau).
* **Validació Quantitativa (PCA):**
    * L'anàlisi de components principals sobre el conjunt de validació mostra una separació clara entre classes.
    * **Distància Euclidiana entre centroides (Sa vs. Pneumònia):** **27.57**.
* **Conclusió:** Aquesta arquitectura resol el problema de continuïtat de la versió amb la U-Net i augmenta dràsticament la interpretabilitat de la patologia. Candidata a model definitiu per a l'extracció de latents.

### 09/02/26 - v7_Latent_Flow_Matching_09022026.ipynb

## Flow Matching sobre l’Espai Latent

Després de consolidar el **Spatial VAE** com a extractor de representacions contínues, s'introdueix el mòdul de **Flow Matching** per modelar la dinàmica entre estats latents. Aquesta fase transforma de l'autoencoder reconstructiu a un **model generatiu dinàmic** capaç de simular trajectòries entre estats clínics, en el nostre cas, pneumonies.

**Objectiu:** Aprendre el **camp de velocitat** que transforma un latent de pacient sa en un latent de pneumònia dins l’espai latent del VAE.  
En lloc de generar imatges directament, el model aprèn:

$$dz/dt = v(z, t)$$

És a dir, com es mou un punt dins l'espai latent al llarg del temps.

---


## 1. Construcció del Dataset de Latents

* **Canvi:** Creació d’un dataset específic per entrenar el Flow.
* **Contingut:** Parelles `(z, label)`.

El model necessita parelles de trajectòria:
* $z_0$: Latent de pacient **SA**.
* $z_1$: Latent de pacient **PNEUMÒNIA**.

Aquestes parelles defineixen l’inici i el final de la trajectòria de transformació.

---

## 2. Interpolació temporal

Per entrenar el model no es simula tota la trajectòria, sinó punts intermedis aleatoris. S'utilitza la interpolació:
$$z_t = (1 - t) \cdot z_0 + t \cdot z_1$$

on:
* $t$ entre 0 i 1 és un instant de temps aleatori.
* $z_t$ és un estat intermedi del procés patològic.

Això permet entrenar el model amb un únic pas de regressió.

---

## 3. Objectiu d’entrenament (Velocity Matching)

El model aprèn la velocitat que connecta els dos estats:
$$real = z_1 - z_0$$

La **loss** força que el model compleixi:
$$v(z_t, t) \approx z_1 - z_0$$

$$Loss = || v_{pred} - (z_1 - z_0) ||^2$$

Aquest pas converteix el problema en l'aprenentatge d’un **camp vectorial continu**.

---

## 4. Model del Camp Vectorial (Flow CNN)

* **Arquitectura:** CNN aplicada sobre el latent espacial.
* **Entrada del model:** `input = concat(z, t)`.

El temps es converteix en un mapa espacial i s’afegeix com a canal extra. 

**Interpretació:** El model respon a  la pregunta: *"Si estic en aquest punt del latent i en aquest instant temporal, cap a on m’he de moure?"*

---

## 5. Integració ODE (Generació de Trajectòries)

Un cop entrenat el camp vectorial, es pot generar la trajectòria completa integrant l'Equació Diferencial Ordinària (ODE). 

**Integrador utilitzat:** Euler
$$z(t + \Delta t) = z(t) + v(z, t) \cdot \Delta t$$

Aquest procés genera múltiples latents intermedis entre l'estat sa i el de pneumònia.

## Resultat

S’ha generat la primera trajectòria completa:

![Trajectòria Generada](/assets/trajectoria_FM_v1.png)



### 11/02/26 - Addition of metric learning and Geodesic correction training

We learn a radial basis function metric in the latent space.
The metric works by fitting k-means to the latents of the dataset, and then learning distances from the k-means clusters as belonging to the latent distribution.

Once the metric is learnt it is used to train a geodesic correction of the interpolants calculation such that newly computed
interpolants will fall close to the latent distribution.


### 12/02/26 - Addition of multi-stage autoencoder

This autoencoder reconstructs from a latent of size 98 using interpolated upscaling.
Interpolations allow to grow feature maps in fractional steps. With more steps we may achieve
better reconstruction quality.

Another novelty is that it uses Convolutional Block Attention Modules (https://www.digitalocean.com/community/tutorials/attention-mechanisms-in-computer-vision-cbam) which should allow the model to focus on more relevant parts of the feature maps for reconstruction.

The mask during training was 0.5. This time a learned parameter is used for the mask.
The encoder embedding is also updated for RoPE.

Finally, it is trained using Perceptual Similarity loss  on top of L1 distance.
(https://github.com/richzhang/PerceptualSimilarity)


### 13/02/26 - Crear fitxer train Metric Flow Matching i usar latents existents

Codi que ajunta el MFM del Albert amb les latents (antigues en vectors i 256 Dim) de la Sandra. Ja que el codi assumeix els inputs de la RBFMetric com a flat vectors.
Un cop funcioni millorar a usar les latents de features map (`latents_pure_train.npy`).

Canvis al codi respecte el codi de metric learning and Geodesic correction training:

- Afegir load de les latents, en comptes d’usar autoencoder.
- Treure del inici: torch.set_default_dtype(torch.float16)
- Afegir guardar correctament part de metrica i poder fer load de tot.
- Fer un metric sanity check al final
- loss.backward(retain_graph=True)   treure el retain_graph, gasta mes memòria
- El W en el metric no posar-lo float 16. Ara tot fp32, sinó podia donar mala resolució de gradients i l'aprenentatge dels pesos inestable
- Cambiar el clamp W a softplus parametrization. Es força valors positius i es normalitza, així l'output és (0, 1]
- Fix dels index en el compute_cluster_points_indexes, abans retornava labels
- Treure el clamp i 1 - abs(1-metric)
- Afegir parametre eps=1e-8 (deixar-lo per defecte) no és gaire rellevant
- Actualitzar la logica del forward a la classe RBFMetric
- Afegir nous atributs de: cluster_sizes...

Estat actual: s’ha trobat un outlier amb una lambda, i els kernels es fan estrets i fa underflow a 0.  (Cosa a tenir en compte)

Queda pendent alguns fixes per entrenar Gamma i el Vector Field.

### 14/02/2026 - Refactorització a Spatial VAE High-Res (v2.0)

## Noves Funcionalitats i Arquitectura:

- Transició a VAE Pur (Zero Skips): Eliminació completa de les skip connections tipus U-Net. Ara l'arquitectura és un "bottleneck" real que força el model a comprimir tota la informació semàntica dins l'espai latent.

- Alta Resolució Latent ($14 \times 14$): Modificació de l'extracció de característiques de la ResNet18 (tallant a la Layer 3 en lloc de l'última) per augmentar el mapa espacial de $4 \times 7 \times 7$ a $4 \times 14 \times 14$. Això preserva millor les formes detallades de la patologia.

- Capa Sigmoid Final: Inclusió d'una capa Sigmoid a la sortida del Decoder per forçar que la imatge generada estigui estrictament en el rang normatiu $[0, 1]$, evitant aberracions de contrast (píxels grisos) en interpolacions i reconstruccions.

## Funcions de Pèrdua (Loss) - Idea de l'Albert -

VGG16 Perceptual Loss: Substitució de l'error MSE clàssic per una combinació de L1 Loss + VGG Perceptual Loss (amb normalització d'ImageNet). Això penalitza la pèrdua de textures i elimina l'efecte "borrós" típic dels VAEs, generant vores i costelles molt més nítides. 

## Optimització d'Entrenament (GPU)

-Gestió de Memòria: Implementació de Gradient Accumulation i Automatic Mixed Precision (AMP) (torch.cuda.amp). Això permet simular Batch Sizes grans (ex: 32) processant paquets petits, adaptant el model complex de $224 \times 224$ a GPUs de 14GB sense trencar la memòria.

## Validació i Extracció de Dades

- Sanity Check Espacial ("Frankenstein Test"): Validació superada. Es va demostrar que l'espai latent manté la topologia 2D fusionant la meitat d'un latent "Sa" amb la meitat d'un latent "Pneumònia", resultant en una imatge reconstruïda meitat sana/meitat malalta.
  
- Extracció Clean Data: Generació i guardat amb èxit dels nous conjunts de latents definitius (latents_train_VAE_14_02_2026.npy i labels_train_VAE_14_02_2026.npy, també em guardo els corresponents als de validació) amb dimensió (N, 4, 14, 14), llestos per ser usats com a entrada baseline per al mòdul de Flow Matching.


## 2026-02-21: VAE High-Res Upgrade & version 2 TorchXRayVision Integration

### 1. Custom VAE High-Res ($4 \times 28 \times 28$)    
En aquesta primera iteració, es va optimitzar l'arquitectura base (ResNet18) per maximitzar la retenció d'estructures anatòmiques fines, sacrificant un nivell de compressió a favor de la fidelitat visual.

* **Script d'Entrenament:** `v9_vae_28x28_res.py` 
* **Directori d'Artefactes:** `./artifacts_final/`
* **Pesos del Model:** * Model complet: `vae_final_sigmoid.pth`    
    * Només Decoder: `decoder_only_final.pth`   Només em deixa pujar aquest!!!! per limit memòria
* **Matrius de Latents (Input per a Flow Matching):**
    * Train: `latents_train_v9.npy` i `labels_train_v9.npy`     
    * Val: `latents_val_v9.npy` i `labels_val_v9.npy`           
* **Ajust d'Arquitectura:** Ampliació del coll d'ampolla (*bottleneck*) dinàmic. L'espai latent va passar de $4 \times 14 \times 14$ a una resolució de $4 \times 28 \times 28$ (3.136 dimensions).
* **Epochs:** VAE entrenat amb 100 epochs.
* **Impacte Visual:** Millora en la nitidesa anatòmica. Les estructures òssies i les vores cardíaques es reconstrueixen sense artefactes de borrositat.
* **Espai Latent (PCA):** Separació basal estructurada entre classes aconseguint una distància euclidiana entre centroides de **26.51**.

### 2. Clinical Expert VAE (Transfer Learning amb TorchXRayVision)
En la segona iteració, es va substituir l'*encoder* genèric per un model expert en domini mèdic per capturar característiques clínicament rellevants (patologies) en lloc de simples textures.

* **Script d'Entrenament:** `v10_train_vae_xrv.py` 
* **Directori d'Artefactes:** `./artifacts_xrv/`
* **Pesos del Model:** * Model complet: `vae_xrv_final.pth`    
    * Només Decoder: `decoder_xrv_final.pth`
* **Matrius de Latents (Input per a Flow Matching):**
    * Train: `latents_xrv_train.npy` i `labels_xrv_train.npy`  
    * Val: `latents_xrv_val.npy` i `labels_xrv_val.npy`        
* **Cirurgia de Model:** Integració del model `XRV-ResNetAE-101-elastic` de la llibreria `torchxrayvision`. Extracció de tensors de 512 canals mantenint la resolució de $28 \times 28$.
* **Entrenament Bifàsic:** 20 *epochs* de *Warm-up* (encoder congelat) + 60 *epochs* de *Fine-Tuning* (descongelació total, LR `1e-5`). Convergència per *Early Stopping* a l'epoch 80 inicialment s'havia configurat per entrenar amb 100 *epochs*.
* **Fites de Rendiment:**
    * **Loss L1:** **0.0197**.
    * **Salt qualitatiu en PCA:** La distància entre centroides va augmentar a **30.97** (un increment substancial en la separabilitat de la patologia).
    * **Salut de l'Espai Latent:** Distància mitjana entre veïns de **57.78**, assegurant una distribució àmplia i rica en variància.

### Millores Generals del Pipeline
* **Early Stopping Automàtic:** Integrat amb una paciència de 15 *epochs* per evitar *overfitting* i optimitzar el temps de còmput.
* **Mètriques Desacoblades:** Intent sense èxit d'integració de TensorBoard mesurant independentment *L1 Pixel Loss*, *VGG Perceptual Loss* i *KL Divergence*.

Compartit al drive els arxius esmentats necessaris perquè ara ja pesen molt. Quan tingui el codi net, sí que el pujaré aquí al GitHub.


## 26/02/2026: Acabar MFM flat vector latent unpaired patients
El codi: `v2_training_metric_flow_matching_26022026.ipynb` conté com a objectiu final l'entrenament del vector field.

Entrenar una mètrica (RBFMetric) alta quan z és a prop del manifold/distribució de latents. A partir d’aquest score construïm el cost de la mètrica. La Gamma s’entrena usant la metrica entrenada i aprèn una correcció, que fa corbar la línia recta entre punts finals, que fa que siguí un camí que es manté a prop del manifold. El vector field usa la gamma pel training i acaba sent la velocity que el flow ha de seguir. El resultat és el learned vector field que usa el codi del Sergi.

Aquesta versió 2 usa unpaired patients (no fem endpoint-conditioned) i usa flat vectors. Inicialment es prova amb el multi-stage autoencoder de l’Albert de 98 dim el flat vector de latents.

Canvis al Vector Field:

Abans fèiem que el timestep_embed agrupés tota la info del temps, ara ho fa en hidden_dim, no en una sola dimensió. I ara es pot combinar amb les latent features en el hidden space.

No aplicar `silu` al output del vector field, pot distorsionar la senyal/magnitud. Deixem l’output lineal i deixem que la velocitat output en espai latent sigui positiva o negativa sense restringir la magnitud.

Traiem `timesteps = linspace(...)` feia que estigui lligat al nombre d’imatges.


Paper del Riemannian metric que considera la geometria (https://arxiv.org/pdf/2008.00565)

## 27/02/2026: Inference RK4 sobre MFM Flat 98D + Visualització de Trajectòria

Després de completar l’entrenament del Metric Flow Matching (MFM) unpaired en espai latent flat 98D, s’implementa la fase completa d’inferència ODE amb integrador d’ordre superior i visualització clínica de la progressió.

Notebook associat: `v3_training_metric_flow_matching_with_generation_26022026.ipynb`

Objectiu: Generar una trajectòria latent contínua partint d’un pacient sa real, integrant el vector field entrenat amb MFM i reconstruint els estats intermedis mitjançant el multi-stage autoencoder (latent 98D).

Model utilitzat:
* Vector Field entrenat amb MFM unpaired
* Autoencoder multi-stage (latent_dim=98)
* Integració ODE amb Runge-Kutta 4 (RK4)

Pipeline inferencia: 

Healthy Image -> Encoder (AE 98D) -> z0 -> RK4 Integration amb VectorField MFM -> {z_t} trajectòria completa -> Decoder multi-stage -> Seqüència d’imatges (progressió)

S’ha generat la trajectòria completa des de t=0 fins t=1 amb 80 passos RK4. Imatge guardada a `/assets/MFM_RK4_flat98.png`

Proper Pas - Migrar MFM a:
* Latents espacials (4×28×28)
* Vector Field tipus U-Net
* Flow Matching sobre tensors espacials

## 28/02/2026 : Millores Spatial VAE amb model Torchxrayvision (Nitidesa i Estabilitat Numèrica)

* **Transició a BCEWithLogitsLoss:** Es va substituir la mètrica de pèrdua de píxel original (L1 Loss) per *Binary Cross Entropy*. Aquest canvi aprofita l'escala logarítmica per penalitzar severament els errors de contrast, millorant dràsticament la definició de vores anatòmiques complexes (costelles, silueta cardíaca) i la nitidesa de les opacitats.
  
* **Optimització per a Mixed Precision (AMP):** Per evitar inestabilitats numèriques i infinits al calcular la BCE en 16-bits, es va eliminar la capa `Sigmoid` final del *Decoder*. Ara el model escup *logits* purs, fusionant l'activació i la pèrdua en una única operació segura (`BCEWithLogitsLoss`).
  
* **Validació de Continuïtat (Interpolació Latent):** S'ha dissenyat i executat un experiment d'interpolació lineal a l'espai latent de $4 \times 28 \times 28$. Els resultats demostren una transició fluida i anatòmicament coherent entre el centroide "Sa" i el centroide "Pneumònia", descartant l'existència de zones mortes o col·lapse modal i donant llum verda per a la fase de *Flow Matching*.

## 03/03/2026: Entenament MFM amb flat i spatial latents

Crear el fitxer:
v4_training_MFM_flat_and_spatial_unet_03032026

Afegir la variable MODE per controlar si és flat o spatial. Afegir la part de spatial latents (4x28x28) i adaptant el vector field perque usi una U-Net.

He agafat la idea de la mini U-net, pero adaptat, he reduit la profunditat, fer 4 o 5 downsamples era excessiu i acabava col·lapsant massa la resolució espacial, n’he deixat 2. Agrupar les convolucions en blocs residuals(ResBlock) i afegir condicionament temporal.

In the PneumoniaMNIST, the learned spatial latent space appears sufficiently smooth that straight healthy-to-sick interpolations are already assigned similar manifold scores to real data. As a result, the Gamma correction network receives only a weak signal and produces minimal curvature. This suggests that, in this simplified setting, the main benefit comes from the spatial vector field itself, while metric-based geodesic correction may only become useful in more complex latent geometries or richer clinical datasets.

També fent proves hem vist que RK4 és millor pels resultats a nivell numèric a la part del vector field après, que no pas el ODE.

Al no ser endpoint conditioned, els resultats van d’un sà, cap a un malalt qualsevol, però la image final també està generada, no és un malalt existent. I s’ha observat en primer lloc que els resultats de flat vector semblen ser millors.

## 06/03/2026:  **Versió Final:** `VAE_XRV v11` *(Arquitectura amb decoder de 10 passes i Estabilització)*

* **[Added]** Nova arquitectura de Decoder profund de 10 passes a la classe `SpatialVAE_XRV` per optimitzar la definició de les textures anatòmiques.
  
* **[Changed]** Implementació de disseny "híbrid fraccional" al Decoder: s'han intercalat 3 capes d'escalat (`ConvTranspose2d` amb `stride=2`) amb 7 capes de refinament (`Conv2d` amb `padding=1`). Això permet augmentar la resolució de 28x28 a 224x224 sense patir explosions de memòria espacial a la GPU.
  
* **[Changed]** Entrenament consolidat fins a 96 èpoques (amb *Early Stopping*). Millora significativa de les mètriques: VGG Perceptual Loss reduïda a un mínim de **0.0084** (major nitidesa) i Divergència KL estabilitzada a **2.88** (espai latent òptim).
  
* **[Fixed]** Actualitzat el *pipeline* d'inferència i dibuix de gràfics. S'ha afegit l'aplicació explícita de `torch.sigmoid()` sobre les prediccions del model per visualitzar-les correctament, atès que l'arquitectura final no inclou la capa Sigmoide per mantenir la compatibilitat matemàtica amb `BCEWithLogitsLoss`.

> **Notes de Disseny Estratègic:** Es prioritza mantenir la topologia suau, contínua i sense artefactes de l'espai latent actual. Aquest és un requisit matemàtic indispensable per garantir la convergència de les Equacions Diferencials Ordinàries (ODEs) en el model de *Flow Matching* que s'implementarà a la Fase 2.