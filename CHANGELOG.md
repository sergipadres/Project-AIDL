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