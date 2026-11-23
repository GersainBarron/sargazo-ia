"""Inventario de MARIDA: cuántos parches y píxeles hay por clase (énfasis en sargazo).

Clases MARIDA: 1 Marine Debris, 2 Dense Sargassum, 3 Sparse Sargassum, 4 Natural Organic Material,
5 Ship, 6 Clouds, 7 Marine Water, 8 Sediment-Laden Water, 9 Foam, 10 Turbid Water,
11 Shallow Water, 12 Waves, 13 Cloud Shadows, 14 Wakes, 15 Mixed Water.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW  # noqa: E402

NOMBRES = {1: "Marine Debris", 2: "Dense Sargassum", 3: "Sparse Sargassum", 4: "Natural Organic Material",
           5: "Ship", 6: "Clouds", 7: "Marine Water", 8: "Sediment-Laden Water", 9: "Foam", 10: "Turbid Water",
           11: "Shallow Water", 12: "Waves", 13: "Cloud Shadows", 14: "Wakes", 15: "Mixed Water"}

base = DATA_RAW / "marida"
mascaras = sorted((base / "patches").rglob("*_cl.tif"))
pix, parches = Counter(), Counter()
con_sargazo = []
for m in mascaras:
    with rasterio.open(m) as r:
        a = r.read(1)
    vals, cnt = np.unique(a[a > 0], return_counts=True)
    for v, c in zip(vals, cnt):
        pix[int(v)] += int(c); parches[int(v)] += 1
    if np.isin(a, [2, 3]).any():
        con_sargazo.append(m.name.replace("_cl.tif", ""))

escenas = {m.parent.name for m in mascaras}
print(f"Parches: {len(mascaras)} | escenas Sentinel-2: {len(escenas)}")
print(f"Parches con sargazo (clase 2 o 3): {len(con_sargazo)}\n")
print(f"{'clase':<26}{'parches':>8}{'píxeles':>10}")
for k in sorted(pix):
    print(f"{k:>2} {NOMBRES.get(k, '?'):<23}{parches[k]:>8}{pix[k]:>10}")
for s in ("train", "val", "test"):
    ids = (base / "splits" / f"{s}_X.txt").read_text().split()
    n = sum(1 for i in ids if any(c.endswith(i) or i in c for c in con_sargazo))
    print(f"split {s}: {len(ids)} parches, {n} con sargazo")
(base.parent.parent / "processed").mkdir(exist_ok=True)
(base.parent.parent / "processed" / "marida_parches_con_sargazo.txt").write_text("\n".join(con_sargazo))
