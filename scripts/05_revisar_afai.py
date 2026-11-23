"""Revisa los NetCDF de AFAI descargados y genera un mapa de frecuencia de sargazo.

Uso:
    python scripts/05_revisar_afai.py --producto 7D
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW, RAIZ  # noqa: E402

# Umbral aproximado de AFAI para sargazo flotante (a calibrar; la USF usa ~0.0002–0.001 en reflectancia).
UMBRAL = 0.0005


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--producto", default="7D")
    a = p.parse_args()

    archivos = sorted((DATA_RAW / "afai" / a.producto).glob("*.nc"))
    ds = xr.concat([xr.open_dataset(f) for f in archivos], dim="time")  # sin dask
    afai = ds["AFAI"]
    print(ds)
    print(f"\nFechas: {str(afai.time.values[0])[:10]} → {str(afai.time.values[-1])[:10]} ({afai.sizes['time']} pasos)")

    for f in archivos:
        d = xr.open_dataset(f)["AFAI"]
        validos = float(np.isfinite(d).mean()) * 100
        sarg = float((d > UMBRAL).sum(("latitude", "longitude")).mean())
        print(f"{f.name}: {d.sizes['time']} días | {validos:5.1f}% píxeles válidos | ~{sarg:,.0f} píxeles/día > umbral")

    freq = (afai > UMBRAL).mean("time")
    fig, ax = plt.subplots(figsize=(9, 8))
    freq.where(freq > 0).plot(ax=ax, cmap="YlOrBr", vmin=0, vmax=0.3,
                              cbar_kwargs={"label": "fracción de días con AFAI > umbral"})
    ax.set_title(f"Frecuencia de sargazo (AFAI {a.producto}) · mar–ago 2025\nGersain Barrón · datos NOAA/USF")
    ax.set_facecolor("#dfe8ee")
    salida = RAIZ / "notebooks" / "figuras" / f"frecuencia_afai_{a.producto}_2025.png"
    fig.savefig(salida, dpi=120, bbox_inches="tight")
    print(f"\nMapa: {salida}")


if __name__ == "__main__":
    main()
