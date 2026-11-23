# 05 · Simulador web (sección Lab de gersadesign.com)

## Idea
Un mapa 3D del Caribe mexicano (península de Yucatán en relieve, mar animado) donde:
- Un **control deslizante de fecha** reproduce la temporada (manchas detectadas por la red, en dorado).
- Un botón **"Predecir 7 días"** muestra la predicción de la ConvLSTM encima, con su incertidumbre (halo translúcido).
- Flechas de **corrientes y viento** animadas (partículas).
- **Semáforo por tramo de costa** (Cancún, Puerto Morelos, Playa del Carmen, Tulum, Mahahual…) con la probabilidad de arribo.
- Panel "**Cómo funciona**": diagrama U-Net → ConvLSTM, métricas contra los baselines y créditos de los datos.
- Modo "**Compara**": persistencia vs física vs red neuronal, lado a lado.

## Técnica
- Three.js; relieve desde un DEM público (SRTM/GEBCO) simplificado a un mesh ligero.
- Datos precalculados en JSON/PNG (máscaras como texturas por día), servidos desde la API del sitio (Hostinger) o como archivos estáticos.
- Bilingüe ES/EN, estilo coherente con la identidad "fieltro" del sitio.
- Firmado por Gersain Barrón; leyenda: "Propuesta independiente con datos abiertos de NOAA/USF, Copernicus y ERA5".

## Datos que debe exportar el pipeline
```json
{
  "fecha": "2025-06-15",
  "deteccion": "tiles/2025-06-15/mask.png",
  "prediccion": [{"t": 1, "mask": "...png", "sigma": "...png"}, "..."],
  "costa": [{"tramo": "Tulum", "p_arribo": 0.82}, "..."],
  "corrientes": "fields/2025-06-15/uv.bin"
}
```
