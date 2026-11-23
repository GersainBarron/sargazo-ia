"""Calibra el baseline de advección: factor sobre la corriente y windage, evaluado en 2024 (validación)."""
import importlib.util
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cl", RAIZ / "scripts" / "10_entrenar_convlstm.py")
cl = importlib.util.module_from_spec(spec); spec.loader.exec_module(cl)

ds = cl.Temporadas([2024])
d = ds.t[0]
H = cl.HORIZONTE


def adv(t, fc, w):
    fut = slice(t + 1, t + H + 1)
    u = fc * d["uo"][fut].mean(0) + w * d["u10"][fut].mean(0)
    v = fc * d["vo"][fut].mean(0) + w * d["v10"][fut].mean(0)
    seg = H * 86400
    dlat = float(np.diff(d["lat"]).mean()); dlon = float(np.diff(d["lon"]).mean())
    di = v * seg / 111e3 / dlat
    dj = u * seg / (111e3 * np.cos(np.deg2rad(d["lat"]))[:, None]) / dlon
    ii, jj = np.meshgrid(np.arange(u.shape[0]), np.arange(u.shape[1]), indexing="ij")
    return cl.map_coordinates(np.nan_to_num(d["frac"][t]), [ii - di, jj - dj], order=1, mode="nearest")


muestras = []
for k in range(0, len(ds), 3):
    _, t = ds.items[k]
    _, _, y, m, _ = ds[k]
    y, m = y.numpy(), m.numpy()
    if m.sum() >= 100:
        muestras.append((t, y, m))

base = np.mean([cl.metricas(np.nan_to_num(d["frac"][t]), y, m)[0] for t, y, m in muestras])
print(f"Persistencia MSE {base:.5f} (n={len(muestras)})")
import sys
FACTORES = [float(v) for v in sys.argv[1:]] or [0.0, 0.25, 0.5, 1.0]
for fc in FACTORES:
    for w in (0.0, 0.01, 0.02, 0.03):
        mse = np.mean([cl.metricas(adv(t, fc, w), y, m)[0] for t, y, m in muestras])
        iou = np.mean([cl.metricas(adv(t, fc, w), y, m)[1] for t, y, m in muestras])
        print(f"corriente×{fc:<4} windage {w:.2f} → MSE {mse:.5f} skill {1 - mse / base:+.3f} IoU {iou:.3f}", flush=True)
