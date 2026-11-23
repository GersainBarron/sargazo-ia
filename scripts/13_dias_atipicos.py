"""Detecta días atípicos en la cobertura diaria de sargazo (AFAI 7D) y genera una lista de exclusión.

Criterio: cobertura del día > 2.5× o < 1/2.5 de la mediana móvil centrada de 15 días.
Salida: data/processed/convlstm/dias_atipicos.json y notebooks/figuras/serie_cobertura.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "processed" / "convlstm"
FACTOR = 2.5

atipicos = {}
fig, axs = plt.subplots(7, 1, figsize=(14, 16), sharey=False)
for ax, y in zip(axs, range(2019, 2026)):
    z = np.load(DATOS / f"temporada_{y}.npz", allow_pickle=True)
    frac = z["frac"].astype(np.float32); mar = z["mar"]
    cov = pd.Series(np.nanmean(frac[:, mar], 1) * 100, index=pd.to_datetime(z["fechas"]))
    med = cov.rolling(15, center=True, min_periods=5).median()
    malo = (cov > FACTOR * med) | (cov < med / FACTOR)
    atipicos[str(y)] = [d.strftime("%Y-%m-%d") for d in cov.index[malo]]
    ax.plot(cov.index, cov, lw=1, label="cobertura diaria")
    ax.plot(med.index, med, lw=1.5, ls="--", label="mediana móvil 15 d")
    ax.scatter(cov.index[malo], cov[malo], c="red", s=18, zorder=3, label="atípico")
    ax.set_ylabel(f"{y}\n% mar"); ax.grid(alpha=.3)
    print(f"{y}: {malo.sum()} días atípicos de {len(cov)}: {', '.join(atipicos[str(y)][:12])}{' …' if malo.sum() > 12 else ''}")
axs[0].legend(loc="upper left")
fig.suptitle("Cobertura diaria de sargazo (AFAI 7D, mar adentro) · Gersain Barrón · datos NOAA/USF")
fig.savefig(RAIZ / "notebooks" / "figuras" / "serie_cobertura.png", dpi=90, bbox_inches="tight")
json.dump(atipicos, open(DATOS / "dias_atipicos.json", "w"), indent=1)
