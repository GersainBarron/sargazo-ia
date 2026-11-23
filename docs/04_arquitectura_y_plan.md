# 04 · Arquitectura y plan de trabajo

## Arquitectura

```
          ┌────────────────────── DETECCIÓN ──────────────────────┐
AFAI 1D/7D (1.6 km) ─┐                                             │
Sentinel-3 OLCI ─────┼─► Preproceso (máscara nubes/glint, recorte) ─► U-Net ─► máscara diaria de sargazo
Sentinel-2 MSI ──────┘   (MARIDA para entrenar)                    │      (probabilidad 0–1, rejilla común)
          └────────────────────────────────────────────────────────┘
                                    │
          ┌────────────────────── PREDICCIÓN ─────────────────────┐
máscaras t-6…t ─┐                                                   │
corrientes uo,vo ┼─► tensor (T, C, H, W) ─► ConvLSTM / U-Net temporal ─► máscaras t+1…t+7
viento u10,v10 ──┘                                                  │
          └────────────────────────────────────────────────────────┘
                                    │
               Riesgo por tramo de costa (probabilidad de arribo, toneladas estimadas)
                                    │
             Airflow (diario) ─► API del sitio ─► Simulador Three.js (Lab de gersadesign.com)
```

### Rejilla común
- Reproyectar todo a la rejilla AFAI (0.015°) en la zona lat 14–24, lon −90 a −80 → ~667 × 667 celdas.
- Para entrenar la ConvLSTM, parches de 128×128 o reducir a 0.03°.

### Modelos
1. **Detección**
   - Baseline: umbral AFAI (método clásico de la USF).
   - Modelo: U-Net (encoder ResNet34 o EfficientNet preentrenado, librería `segmentation_models_pytorch`).
   - Entrada Sentinel-2: bandas B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 + índices FAI/AFAI/NDVI.
   - Pérdida: Dice + BCE focal (clase muy desbalanceada).
2. **Predicción**
   - Baseline 1: persistencia (mañana = hoy).
   - Baseline 2: advección lagrangiana con OpenDrift (corrientes + windage ~1–3 %, a calibrar).
   - Modelo: ConvLSTM (o SimVP/U-Net temporal) con entrada de 7 días [máscara, uo, vo, u10, v10] → salida de 7 días.
   - Extra: modelo híbrido = advección física + red que aprende el residuo.

### Métricas
- Detección: F1, IoU, precisión/recall contra MARIDA y etiquetas propias. Meta: F1 ≥ 0.85 en S2 (referencia: Laval 2023, F1 0.88).
- Predicción: IoU por horizonte (t+1…t+7), distancia entre centroides, *hit rate* de arribo por tramo de costa. Siempre contra persistencia y advección.
- Validación en tierra: registros de recolección de SEMARNAT por playa y fecha.

## Fases

| Fase | Entregable | Duración estimada |
|---|---|---|
| F0 · Entorno y datos | Cuentas creadas, temporada 2025 descargada (AFAI, corrientes, viento), notebook de exploración con mapas | 1 semana |
| F1 · Detección | Baseline AFAI + U-Net entrenada con MARIDA, ajustada a escenas del Caribe mexicano | 1–2 semanas |
| F2 · Predicción | Dataset espaciotemporal, baselines, ConvLSTM entrenada con 2019–2024 y validada con 2025 | 2 semanas |
| F3 · Validación | Comparación contra SEMARNAT/semáforo, reporte de métricas y casos de estudio | 1 semana |
| F4 · Pipeline | DAG de Airflow diario → JSON/teselas → API del sitio | 1 semana |
| F5 · Simulador y publicación | Simulador Three.js en Lab, repo público con README bilingüe, post en LinkedIn | 1 semana |

**MVP para mostrar rápido (≈2 semanas):** AFAI histórico + persistencia/advección + ConvLSTM simple + simulador con la temporada 2025 "reproducida" y la predicción encima.

## Hardware
- Entrenamiento: cualquier GPU NVIDIA (CUDA) o Apple Silicon (PyTorch MPS). Los modelos son pequeños; no requiere nube.

## Riesgos
| Riesgo | Mitigación |
|---|---|
| Nubes y brillo solar dejan huecos en AFAI | Usar el acumulado de 7 días y máscara de datos faltantes como canal de entrada |
| Pocas etiquetas del Caribe mexicano | MARIDA + pseudo-etiquetas AFAI + revisión manual |
| Licencias de datos | Todos los usados son abiertos; citar DOI en el README |
| Parecer un proyecto "oficial" | Leyenda "propuesta independiente con datos abiertos", sin logos institucionales |
