# 02 · Fuentes de datos abiertos

Zona de estudio propuesta (Caribe mexicano + aproximación): **lat 14° a 24° N, lon −90° a −80° E**.
Temporada fuerte: marzo a septiembre.

## A. Detección de sargazo (lo que el modelo "ve")

### A1. USF / NOAA AFAI — Alternative Floating Algae Index ⭐ empezar aquí
- **Qué es:** índice diario derivado de MODIS/VIIRS, calculado por el Laboratorio de Oceanografía Óptica de la USF (Chuanmin Hu). Sensible al sargazo flotante, pero se satura con brillo solar, nubes o aerosoles.
- **Resolución:** ~0.015° (~1.6 km). Desde 2016-06-18.
- **Acceso:** ERDDAP, sin cuenta.
  - 1 día: `noaa_aoml_atlantic_oceanwatch_AFAI_1D` → https://cwcgom.aoml.noaa.gov/erddap/griddap/noaa_aoml_atlantic_oceanwatch_AFAI_1D.html
  - 7 días acumulado: `noaa_aoml_atlantic_oceanwatch_AFAI_7D` → https://cwcgom.aoml.noaa.gov/erddap/griddap/noaa_aoml_atlantic_oceanwatch_AFAI_7D.html
  - Espejo 3 días (UAF): `noaa_aoml_3896_1b40_b974` → https://upwell.pfeg.noaa.gov/erddap/griddap/noaa_aoml_3896_1b40_b974.html
  - Variable: `AFAI` (dimensiones time, latitude, longitude).
- **Licencia:** uso y redistribución libres; no apto para uso legal.
- **Referencia:** Hu, C. et al. (2015). Remote Sens. Environ. 167:229-246. doi:10.1016/j.rse.2015.05.022
- **Script:** `scripts/01_descargar_afai.py`
- **Portal SaWS:** https://optics.marine.usf.edu/projects/saws.html

### A2. Sentinel-2 MSI (10–20 m) — detección fina cerca de costa
- **Acceso:** Copernicus Data Space Ecosystem (cuenta gratuita) → https://dataspace.copernicus.eu · API STAC/OData, o vía `sentinelhub`.
- **Uso:** la U-Net de alta resolución, cerca de playas y arrecifes.
- **Tiles que cubren Q. Roo (MGRS):** por definir en la exploración (ej. 16QDJ, 16QEJ, 16QEH…). Verificar con el buscador del Data Space.

### A3. Sentinel-3 OLCI (300 m) — detección regional diaria
- **Acceso:** Copernicus Data Space.
- **Dataset derivado ya procesado:** detección de sargazo diaria con OLCI (datapaper ESSD 2026, Météo-France/IRD), NetCDF en Odatis, DOI 10.12770/1eb82d09-77ed-4f63-9f03-2c3516a9713d → https://essd.copernicus.org/preprints/essd-2026-589/

## B. Datos etiquetados para entrenar

### B1. MARIDA — Marine Debris Archive ⭐
- 1,381 parches de Sentinel-2 de 256×256 (GeoTIFF) con máscaras por píxel y nivel de confianza. 15 clases, **incluye Sargassum** (además de barcos, espuma, estelas, olas, agua clara y turbia, etc.). Las clases "parecidas" ayudan a bajar falsos positivos.
- Zenodo: https://zenodo.org/records/5151941 · Guía: https://marine-debris.github.io
- Cita: Kikaki K. et al. (2022) PLoS ONE 17(1): e0262247.
- **Script:** `scripts/04_descargar_marida.py`

### B2. MADOS — Marine Debris and Oil Spill
- Extensión de MARIDA con derrames de hidrocarburos (útil para la generalización a otras amenazas marinas).
- Zenodo: https://zenodo.org/records/10664073

### B3. Etiquetas propias (pseudo-etiquetas)
- Umbral AFAI + revisión manual en QGIS sobre escenas Sentinel-2 del Caribe mexicano.
- Validación en tierra: reportes ciudadanos de SATsum-Collect (CONABIO) → https://five.epicollect.net/project/sargazo-en-el-gran-caribe

## C. Física para la predicción (lo que mueve al sargazo)

### C1. Corrientes superficiales — Copernicus Marine ⭐
- Producto: `GLOBAL_ANALYSISFORECAST_PHY_001_024` (Mercator, 1/12°). DOI 10.48670/moi-00016
- Dataset de corrientes diarias: `cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m` · variables `uo`, `vo`
- 6-horario: `cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i`
- La lista de variables del producto incluye **deriva de Stokes por oleaje** (x/y), relevante para el sargazo. Hay que verificar en qué dataset ID viene (`copernicusmarine describe --product-id GLOBAL_ANALYSISFORECAST_PHY_001_024`).
- Acceso: Copernicus Marine Toolbox (`pip install copernicusmarine`, `copernicusmarine login`), cuenta gratuita.
- Para histórico largo: reanálisis `GLOBAL_MULTIYEAR_PHY_001_030`.
- **Script:** `scripts/02_descargar_corrientes.py`

### C2. Alternativa: HYCOM
- La NOAA/AOML encontró que agregar el efecto del viento (*windage*) a HYCOM mejora las predicciones de arribo. https://www.aoml.noaa.gov/tracking-pelagic-sargassum/

### C3. Viento a 10 m — ERA5 ⭐
- Copernicus Climate Data Store, dataset `reanalysis-era5-single-levels`, variables `10m_u_component_of_wind`, `10m_v_component_of_wind`. Horario, 0.25°.
- Requiere cuenta CDS y archivo `~/.cdsapirc` con la API key.
- **Script:** `scripts/03_descargar_viento.py`

## D. Validación y contexto
- Registro de recolección de sargazo SEMARNAT (toneladas por playa y fecha) → enlace en https://sargazo.icmyl.unam.mx/en/Sargassum_Links.php
- Semáforo de la Red de Monitoreo del Sargazo de Quintana Roo (Facebook).
- Visualizador LANOT-UNAM.
- Línea de costa: GSHHG o INEGI (para dividir la costa en tramos y medir arribos).

## Cuentas que hay que crear (gratuitas)
| Servicio | URL | Para |
|---|---|---|
| Copernicus Marine | https://data.marine.copernicus.eu/register | corrientes |
| Copernicus Data Space | https://dataspace.copernicus.eu | Sentinel-2/3 |
| Climate Data Store | https://cds.climate.copernicus.eu | viento ERA5 |
| (ninguna) | ERDDAP NOAA | AFAI |
| (ninguna) | Zenodo | MARIDA |
