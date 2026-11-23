"""Máscara de costa para el baseline AFAI, construida con los propios datos (sin descargas extra).

Se excluyen:
  1. Tierra: píxeles que son NaN en todos los días.
  2. Franja costera: tierra dilatada N píxeles (~1.6 km por píxel).
  3. Rasgos estáticos: píxeles que superan el umbral en más del X % de los días
     (el sargazo se mueve; el fondo somero y la turbidez no).

Uso:
    python scripts/08_mascara_costa.py --buffer 3 --persistencia 0.5
Salida:
    data/processed/mascara_costa_afai.nc  y  notebooks/figuras/frecuencia_afai_7D_2025_limpia.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW, RAIZ  # noqa: E402

UMBRAL = 0.0005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", type=int, default=3)
    ap.add_argument("--persistencia", type=float, default=0.5)
    a = ap.parse_args()

    archivos = sorted((DATA_RAW / "afai" / "7D").glob("afai_7D_2025*.nc"))
    afai = xr.concat([xr.open_dataset(f)["AFAI"] for f in archivos], dim="time").load()

    tierra = afai.isnull().all("time").values
    franja = binary_dilation(tierra, iterations=a.buffer) & ~tierra
    sobre = (afai > UMBRAL)
    validos = afai.notnull().sum("time")
    frec_valida = (sobre.sum("time") / validos.where(validos > 0)).fillna(0).values
    estatico = (frec_valida > a.persistencia) & ~tierra & ~franja
    excluir = tierra | franja | estatico

    m = xr.Dataset(
        dict(tierra=(("latitude", "longitude"), tierra), franja_costera=(("latitude", "longitude"), franja),
             estatico=(("latitude", "longitude"), estatico), excluir=(("latitude", "longitude"), excluir)),
        coords=dict(latitude=afai.latitude, longitude=afai.longitude),
        attrs=dict(buffer_px=a.buffer, persistencia=a.persistencia, umbral_afai=UMBRAL, autor="Gersain Barrón"),
    )
    salida = RAIZ / "data" / "processed" / "mascara_costa_afai.nc"
    salida.parent.mkdir(parents=True, exist_ok=True)
    m.astype("uint8").to_netcdf(salida)

    antes = sobre.where(~xr.DataArray(tierra, dims=("latitude", "longitude"))).sum(("latitude", "longitude")).mean().item()
    despues = sobre.where(~xr.DataArray(excluir, dims=("latitude", "longitude"))).sum(("latitude", "longitude")).mean().item()
    print(f"Franja costera: {franja.sum():,} px | estáticos: {estatico.sum():,} px")
    print(f"Píxeles/día sobre umbral: {antes:,.0f} → {despues:,.0f} ({100 * (1 - despues / antes):.0f}% eliminados)")

    freq = sobre.mean("time").where(~xr.DataArray(excluir, dims=("latitude", "longitude")))
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True)
    for ax, datos, titulo in [(axs[0], sobre.mean("time").where(~xr.DataArray(tierra, dims=("latitude", "longitude"))), "Umbral simple"),
                              (axs[1], freq, "Con máscara de costa y rasgos estáticos")]:
        datos.where(datos > 0).plot(ax=ax, cmap="YlOrBr", vmin=0, vmax=0.3, add_colorbar=ax is axs[1],
                                    cbar_kwargs=dict(label="fracción de días con AFAI > umbral") if ax is axs[1] else None)
        ax.imshow(np.where(tierra, 1, np.nan), extent=[float(afai.longitude.min()), float(afai.longitude.max()),
                  float(afai.latitude.min()), float(afai.latitude.max())], origin="lower", cmap="Greys", vmin=0, vmax=2)
        ax.set_facecolor("#dfe8ee"); ax.set_title(titulo)
    fig.suptitle("Sargazo AFAI 7D · mar–ago 2025 · Gersain Barrón · datos NOAA/USF")
    fig_path = RAIZ / "notebooks" / "figuras" / "frecuencia_afai_7D_2025_limpia.png"
    fig.savefig(fig_path, dpi=110, bbox_inches="tight")
    print(f"Mapa: {fig_path}")


if __name__ == "__main__":
    main()
