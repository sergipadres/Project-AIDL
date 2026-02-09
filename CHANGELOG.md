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




