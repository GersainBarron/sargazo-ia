# Metodología

**Sargazo IA** · Gersain Barrón

Este documento describe, con su matemática, cada paso de la cadena: del dato satelital a la detección, de la detección al pronóstico a 7 días, y de ahí a la evaluación.

---

## 1. Señal satelital: índice AFAI

El *Alternative Floating Algae Index* (Wang & Hu, 2016) mide cuánto sobresale la reflectancia en el borde rojo (748 nm) respecto a una línea base entre el rojo (667 nm) y el infrarrojo cercano (869 nm). La vegetación flotante, como el sargazo, refleja más en el borde rojo que el agua:

$$
\mathrm{AFAI} = R_{rc,748} - R'_{rc,748},\qquad
R'_{rc,748} = R_{rc,667} + \left(R_{rc,869} - R_{rc,667}\right)\frac{748-667}{869-667}
$$

donde $R_{rc,\lambda}$ es la reflectancia corregida por Rayleigh. Se usa el producto acumulado de 7 días de NOAA/USF (0.015°, ~1.6 km), que rellena huecos de nubes.

### Cobertura por celda

La rejilla fina (667×667) se agrega por bloques de $4\times4$ en celdas de ~6.7 km. Para la celda $c$ el día $t$:

$$
f_c(t) = \frac{1}{|V_c(t)|}\sum_{p\in V_c(t)} \mathbb{1}\left[\mathrm{AFAI}_p(t) > \tau\right],\qquad \tau = 5\times10^{-4}
$$

$V_c(t)$ son los píxeles válidos (sin nube y fuera de la máscara $M$). La máscara excluye:

- **Tierra:** píxeles sin dato en todas las fechas.
- **Franja costera:** tierra dilatada 3 píxeles (~5 km), donde el fondo somero y la turbidez imitan al sargazo.
- **Rasgos estáticos:** píxeles que superan $\tau$ en más del 50 % de los días de todas las temporadas. El sargazo se mueve; el fondo no.

Con esa máscara desaparece el 38 % de los píxeles que superaban el umbral mar adentro.

### Días atípicos

La cobertura media diaria $\bar f(t)$ se compara con su mediana móvil centrada de 15 días $\tilde f(t)$. Un día es atípico si

$$
\bar f(t) > 2.5\,\tilde f(t)\quad\text{o}\quad \bar f(t) < \tilde f(t)/2.5
$$

Se marcaron 40 días de 1,454, y se excluyen como entrada o como objetivo del pronóstico.

---

## 2. Detección: U-Net sobre Sentinel-2

**Datos.** MARIDA (Kikaki et al., 2022): 1,381 parches Sentinel-2 de 256×256 px a 10 m, 11 bandas, con 15 clases anotadas por píxel. Las clases *Dense Sargassum* y *Sparse Sargassum* forman la clase positiva; las otras 13 (espuma, barcos, estelas, agua turbia, agua somera…) la negativa. Los píxeles sin etiqueta se ignoran.

**Modelo.** U-Net con codificador ResNet34 preentrenado en ImageNet, adaptado a 11 canales. Las bandas se normalizan con media y desviación del conjunto de entrenamiento.

**Pérdida.** Entropía cruzada ponderada más Dice, solo sobre píxeles etiquetados $\Omega$:

$$
\mathcal{L} = -\frac{1}{|\Omega|}\sum_{i\in\Omega}\left[w\,y_i\log p_i + (1-y_i)\log(1-p_i)\right]
\;+\; 1-\frac{2\sum_i p_i y_i + 1}{\sum_i p_i + \sum_i y_i + 1},\qquad w = 10
$$

El sargazo es una clase rara (≈5 mil píxeles frente a cientos de miles), por eso el peso $w$ y el sobremuestreo ×5 de los parches que lo contienen. Aumentación: rotaciones de 90° y volteos.

**Entrenamiento.** AdamW ($\eta=3\times10^{-4}$, decaimiento $10^{-4}$), coseno, 40 épocas. El umbral de decisión (0.60) se elige en validación.

---

## 3. Pronóstico: ConvLSTM

### Entradas

Para pronosticar el día $t+7$ se usa la historia $t-6,\dots,t$, con 6 canales por día:

$$
\mathbf{X}_k = \big[f,\; v,\; u_o,\; v_o,\; u_{10}/10,\; v_{10}/10\big]_k,\qquad k=t-6,\dots,t
$$

- $f$: cobertura.
- $v$: fracción válida.
- $(u_o, v_o)$: corriente superficial de Copernicus Marine (análisis y pronóstico, o reanálisis GLORYS12).
- $(u_{10}, v_{10})$: viento a 10 m de ERA5.

Se añade también el forzamiento medio de la semana siguiente:

$$
\mathbf{F} = \frac{1}{7}\sum_{k=t+1}^{t+7}\big[u_o,\; v_o,\; u_{10}/10,\; v_{10}/10\big]_k
$$

### Celda ConvLSTM (Shi et al., 2015)

Con $*$ convolución 3×3 y $\odot$ producto elemento a elemento:

$$
\begin{aligned}
[\,\mathbf{i}_k;\mathbf{f}_k;\mathbf{o}_k;\mathbf{g}_k\,] &= \mathbf{W} * [\,\mathbf{E}(\mathbf{X}_k);\,\mathbf{H}_{k-1}\,] + \mathbf{b}\\
\mathbf{C}_k &= \sigma(\mathbf{f}_k)\odot\mathbf{C}_{k-1} + \sigma(\mathbf{i}_k)\odot\tanh(\mathbf{g}_k)\\
\mathbf{H}_k &= \sigma(\mathbf{o}_k)\odot\tanh(\mathbf{C}_k)
\end{aligned}
$$

$\mathbf{E}$ es un codificador de dos convoluciones. Hay dos capas ConvLSTM apiladas (48 canales). La cabeza recibe $[\mathbf{H}_t;\,\mathbf{F};\,f_t]$.

### v1: cobertura (regresión residual)

$$
\hat f_{t+7} = \mathrm{clip}\big(f_t + \Delta_\theta(\cdot),\,0,\,1\big),\qquad
\mathcal{L} = \frac{1}{|\Omega|}\sum_{c\in\Omega}\big(\hat f_c - f_c(t+7)\big)^2
$$

El MSE premia predecir un campo suave y de baja amplitud, que difumina los parches.

### v2: probabilidad de presencia (la que usa el simulador)

Objetivo binario $y_c = \mathbb{1}[f_c(t+7) > 0.05]$, salida $p_c = \sigma(z_c)$. Pérdida focal (Lin et al., 2017) más Dice:

$$
\mathcal{L}_{\text{focal}} = -\frac{1}{|\Omega|}\sum_{c}\alpha_c\,(1-p_{t,c})^{\gamma}\log p_{t,c},\qquad
p_{t,c}=\begin{cases}p_c & y_c=1\\ 1-p_c & y_c=0\end{cases},\quad \alpha=0.75,\;\gamma=2
$$

La pérdida focal baja el peso de los casos fáciles (mar abierto vacío) y concentra el aprendizaje en los bordes de las manchas. Se entrena con paro temprano (paciencia 5), y el umbral (0.75) se elige en validación.

**Partición temporal.** Entrenamiento 2019–2023 (941 muestras), validación 2024 (161), prueba 2025 (149). El modelo nunca ve la temporada de prueba.

---

## 4. Métodos de referencia

**Persistencia.** Supone que dentro de 7 días el sargazo estará igual que hoy: $\hat f_{t+7} = f_t$.

**Advección semilagrangiana.** Transporta el campo de hoy con la corriente media más una fracción $w$ del viento (*windage*):

$$
\hat f_{t+7}(\mathbf{x}) = f_t\big(\mathbf{x} - (\bar{\mathbf{u}}_o + w\,\bar{\mathbf{u}}_{10})\,\Delta t\big),\qquad \Delta t = 7\ \text{días}
$$

**Trayectorias lagrangianas.** Se siembran partículas en cada celda con sargazo y se integran con paso de 6 h:

$$
\frac{d\mathbf{x}}{dt} = \mathbf{u}_o(\mathbf{x},t) + w\,\mathbf{u}_{10}(\mathbf{x},t)
$$

Las partículas que tocan costa quedan varadas. Su densidad final es el pronóstico. El *windage* se calibró en validación sobre $w\in\{0,1,2,3\}\,\%$, y el mejor fue $w=0$.

---

## 5. Métricas

$$
\mathrm{IoU} = \frac{|P\cap Y|}{|P\cup Y|},\qquad
F_1 = \frac{2\,TP}{2\,TP+FP+FN},\qquad
\text{skill}_{\text{MSE}} = 1-\frac{\mathrm{MSE}_{\text{modelo}}}{\mathrm{MSE}_{\text{persistencia}}}
$$

- **IoU:** se calcula por día sobre las celdas válidas de mar y luego se promedia.
- **Alerta por tramo de costa:** para Cancún–Isla Mujeres, Puerto Morelos–Playa del Carmen–Cozumel, Tulum–Sian Ka'an y Costa Maya–Mahahual, hay "arribo" si al menos 2 celdas de la franja de ~20 km frente a la costa tienen presencia.

---

## Referencias

- Wang, M. & Hu, C. (2016). Mapping and quantifying *Sargassum* distribution and coverage in the Central West Atlantic using MODIS observations. *Remote Sensing of Environment*, 183, 350–367.
- Hu, C. et al. (2015). Spectral and spatial requirements of remote measurements of pelagic *Sargassum* macroalgae. *Remote Sensing of Environment*, 167, 229–246.
- Arellano-Verdejo, J., Lazcano-Hernández, H. E. & Cabanillas-Terán, N. (2019). ERISNet: deep neural network for *Sargassum* detection along the coastline of the Mexican Caribbean. *PeerJ*, 7, e6842.
- Laval, M. et al. (2023). Detection of *Sargassum* from Sentinel Satellite Sensors Using Deep Learning Approach. *Remote Sensing*, 15(4), 1104.
- Kikaki, K. et al. (2022). MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data. *PLoS ONE*, 17(1), e0262247.
- Shi, X. et al. (2015). Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. *NeurIPS*.
- Ronneberger, O., Fischer, P. & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.
- Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV*.
- Marsh, R. et al. (2021). Seasonal predictions of holopelagic *Sargassum* across the tropical Atlantic: the SARTRAC Ensemble Forecast System. *Frontiers in Marine Science*, 8, 722524.
