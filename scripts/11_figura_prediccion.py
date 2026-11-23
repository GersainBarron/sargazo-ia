"""Figura comparativa de la predicción a 7 días para una fecha de la temporada de prueba (2025).

Paneles: hoy · real en t+7 · persistencia · advección física · ConvLSTM

Uso:
    python scripts/11_figura_prediccion.py --fecha 2025-07-15
    python scripts/11_figura_prediccion.py            # elige el día con más sargazo
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cl", RAIZ / "scripts" / "10_entrenar_convlstm.py")
cl = importlib.util.module_from_spec(spec); spec.loader.exec_module(cl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha")
    ap.add_argument("--v2", action="store_true", help="usar la ConvLSTM v2 (probabilidad de presencia)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = cl.Temporadas([2025])
    d = ds.t[0]
    fechas = [str(f) for f in d["fechas"]]
    if a.fecha:
        t = fechas.index(a.fecha)
        k = ds.items.index((0, t))
    else:
        k = max(range(len(ds)), key=lambda j: np.nansum(d["frac"][ds.items[j][1]]))
        t = ds.items[k][1]

    if a.v2:
        s2 = importlib.util.spec_from_file_location("v2", RAIZ / "scripts" / "14_entrenar_convlstm_v2.py")
        v2 = importlib.util.module_from_spec(s2); s2.loader.exec_module(v2)
        modelo = v2.ConvLSTMv2().to(dev)
        modelo.load_state_dict(torch.load(RAIZ / "modelos" / "convlstm_v2_sargazo.pt", map_location=dev))
    else:
        modelo = cl.SargazoConvLSTM().to(dev)
        modelo.load_state_dict(torch.load(RAIZ / "modelos" / "convlstm_sargazo.pt", map_location=dev))
    modelo.eval()
    x, forz, y, m, _ = ds[k]
    with torch.no_grad():
        pred = modelo(x[None].to(dev), forz[None].to(dev))[0]
        pred = (torch.sigmoid(pred) if a.v2 else pred).cpu().numpy()

    hoy = np.nan_to_num(d["frac"][t]); real = d["frac"][t + cl.HORIZONTE]
    adv = cl.adveccion(d, t, d["lat"])
    mar = d["mar"]; mm = m.numpy()
    paneles = [("Hoy (" + fechas[t] + ")", hoy), ("Real en +7 días (" + fechas[t + cl.HORIZONTE] + ")", real),
               ("Persistencia", hoy), ("Advección física", adv)]
    if a.v2:
        # Presencia: real/persistencia/advección como cobertura; la red como probabilidad (umbral elegido en validación)
        paneles.append(("ConvLSTM v2 · probabilidad", pred * 0.4))  # escala a la barra de color (1.0 → 0.4)
    else:
        paneles.append(("ConvLSTM (red neuronal)", pred))

    ext = [d["lon"][0], d["lon"][-1], d["lat"][0], d["lat"][-1]]
    fig, axs = plt.subplots(1, 5, figsize=(25, 5.6))
    for ax, (tit, campo) in zip(axs, paneles):
        c = np.where(mar, campo, np.nan)
        ax.set_facecolor("#9e9e9e")
        ax.imshow(np.where(mar, 0, np.nan), extent=ext, origin="lower", cmap="Blues", vmin=-1, vmax=1)
        im = ax.imshow(np.where(c > 0.01, c, np.nan), extent=ext, origin="lower", cmap="YlOrBr", vmin=0, vmax=0.4)
        if tit.startswith(("Persistencia", "Advección", "ConvLSTM")):
            if a.v2:
                umbral = 0.75 * 0.4 if tit.startswith("ConvLSTM") else 0.02
                tb = (np.nan_to_num(real) > 0.05) & mm; pb = (campo > umbral) & mm
                tit += f"\nIoU presencia {(pb & tb).sum() / max((pb | tb).sum(), 1):.2f}"
            else:
                mse, iou = cl.metricas(campo, np.nan_to_num(real), mm)
                tit += f"\nIoU {iou:.2f} · MSE {mse * 1e3:.2f}e-3"
        ax.set_title(tit, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs, shrink=0.8, label="cobertura de sargazo (fracción de celda)")
    fig.suptitle("Predicción de sargazo a 7 días · Caribe mexicano · Gersain Barrón · datos NOAA/USF, Copernicus, ERA5",
                 fontsize=13)
    salida = RAIZ / "notebooks" / "figuras" / f"prediccion_7d_{fechas[t]}{'_v2' if a.v2 else ''}.png"
    fig.savefig(salida, dpi=100, bbox_inches="tight")
    print(salida)


if __name__ == "__main__":
    main()
