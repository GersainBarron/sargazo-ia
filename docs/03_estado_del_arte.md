# 03 · Estado del arte

## Detección de sargazo con redes neuronales

| Trabajo | Qué hizo | Por qué importa |
|---|---|---|
| **ERISNet** — Arellano-Verdejo, Lazcano-Hernández, Cabanillas-Terán (ECOSUR Chetumal), *PeerJ* 7:e6842, 2019. doi:10.7717/peerj.6842 | CNN + red recurrente sobre píxeles Aqua-MODIS para detectar sargazo flotante y acumulado en la costa de Quintana Roo. ~90% de probabilidad de clasificación. | **Antecedente mexicano directo.** Es la línea base para citar y superar. |
| **Laval et al.**, *Remote Sensing* 15(4):1104, 2023. https://mdpi.com/2072-4292/15/4/1104 | Redes encoder–decoder específicas para Sentinel-2 MSI y Sentinel-3 OLCI. F1 = 0.88 en MSI y 0.76 en OLCI; superaron a ERISNet, U-Net y SegNet genéricas con el mismo entrenamiento. Coautor mexicano: Adán Salazar-Garibay (CentroGeo). | Muestra que adaptar la arquitectura a las bandas del sensor gana. Buen objetivo de métrica. |
| **Wang & Hu** (USF), "Satellite remote sensing of pelagic Sargassum: the power of high resolution and deep learning". https://repository.library.noaa.gov/view/noaa/56090 | U-Net con pesos VGG16 sobre MSI, OLI, WorldView-2 y PlanetScope. ~98% de exactitud en WV-2. | Receta concreta: U-Net + encoder preentrenado. |
| **Wang & Hu**, "Automatic Extraction of Sargassum Features from Sentinel-2 MSI Images", *IEEE TGRS* 59:2579-2597, 2021. | Extracción automática en MSI. | Preprocesamiento y manejo de falsos positivos. |
| **MARIDA** (Kikaki et al., 2022) | Benchmark de Sentinel-2 con 15 clases, incluido sargazo. | Dataset de entrenamiento listo. |

## Predicción de la deriva

| Trabajo | Qué hizo | Por qué importa |
|---|---|---|
| **NOAA AOML** — *windage* en HYCOM. https://www.aoml.noaa.gov/tracking-pelagic-sargassum/ | Derivadores GPS sobre sargazo real y pseudo-sargazo; agregar windage a HYCOM mejora las predicciones de arribo. | Justifica incluir el viento como entrada. Baseline físico. |
| **Beron-Vera & Bonner** — LSTM vs SINDy (AIP Scilight). https://www.aip.org/scilights/modeling-drifting-seaweed-rafts-that-clog-ports-and-beaches | Ecuaciones de movimiento de balsas de sargazo aprendidas de datos con LSTM y SINDy, entrenadas con simulaciones del modelo eBOMB. | Puente entre física y redes neuronales. |
| **eBOMB / Beron-Vera et al.** (arXiv 2410.01468) | Modelo mecanístico: corrientes, viento, interacción dentro de la balsa y atrapamiento en remolinos. | Modelo físico de referencia. |
| **Opendrift + modelo BOMB** (AGU OSM 2020) | Ecuaciones BOMB en OpenDrift para un sistema operativo con corrientes y viento operacionales. | **OpenDrift** (Python) sirve como baseline lagrangiano listo. |
| **SARTRAC-EFS** — Marsh et al., *Front. Mar. Sci.* 8:722524, 2021 | Pronóstico estacional por ensamble variando windage, crecimiento, mortalidad y hundimiento. | Idea de ensambles e incertidumbre para el simulador. |
| **Lesser Antilles** — *Ocean Science* 18:915, 2022 | Predicción de riesgo costero con corrientes Mercator y HYCOM más windage; HYCOM funcionó mejor allí. Sistema por clustering de bajo costo computacional. | Comparar Copernicus vs HYCOM. |
| **Deep learning + múltiples factores** (Cambridge, *Environmental Data Science*) | Redes profundas sobre múltiples factores para la dinámica del movimiento. | Antecedente directo del enfoque ConvLSTM. |
| **Boletines Météo-France** (MOTHY) | Pronósticos a 4 días y tendencia a 2 semanas con detecciones OLCI + deriva MOTHY. | Horizonte operativo realista: 4–7 días. |

## Huecos que este proyecto puede ocupar
1. **Pronóstico aprendido de punta a punta (ConvLSTM) comparado contra la advección física** para el Caribe mexicano en particular.
2. **Salida por tramo de costa** (Cancún, Puerto Morelos, Playa del Carmen, Tulum, Mahahual…) en lugar de solo mapas: es lo que necesita quien decide dónde poner barreras y buques.
3. **Cuantificación de incertidumbre** (ensambles o MC-dropout) que se ve en el simulador.
4. **Todo abierto y reproducible**, con pipeline diario.
