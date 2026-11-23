"""Entrena una ConvLSTM que predice la cobertura de sargazo a 7 días y la compara con dos baselines.

Tarea: con los últimos 7 días (sargazo + corrientes + viento) y el forzamiento medio de los 7 días
siguientes, predecir la cobertura de sargazo en t+7 (AFAI 7D, ventanas sin traslape).

Baselines:
  - Persistencia: el sargazo de dentro de 7 días es el de hoy.
  - Advección física: se transporta el campo de hoy con corriente + 2 % del viento (windage), 7 días.

Partición temporal: entrenamiento 2019–2023, validación 2024, prueba 2025.
Nota: en este hindcast el forzamiento futuro es el análisis real; en operación vendría de un pronóstico.

Uso:
    python scripts/10_entrenar_convlstm.py --epocas 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import map_coordinates
from torch.utils.data import DataLoader, Dataset

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "processed" / "convlstm"
PASADO, HORIZONTE, WINDAGE = 7, 7, 0.02


def cargar(anio):
    z = np.load(DATOS / f"temporada_{anio}.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["frac"] = d["frac"].astype(np.float32)
    d["valid"] = d["valid"].astype(np.float32)
    fechas = np.array(d["fechas"], dtype="datetime64[D]")
    # Índices t con t-6..t y t+7 consecutivos
    idx = [t for t in range(PASADO - 1, len(fechas) - HORIZONTE)
           if (fechas[t + HORIZONTE] - fechas[t - PASADO + 1]).astype(int) == PASADO - 1 + HORIZONTE]
    d["idx"] = idx
    return d


class Temporadas(Dataset):
    def __init__(self, anios):
        self.t = [cargar(a) for a in anios if (DATOS / f"temporada_{a}.npz").exists()]
        self.items = [(i, t) for i, d in enumerate(self.t) for t in d["idx"]]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, t = self.items[k]
        d = self.t[i]
        s = slice(t - PASADO + 1, t + 1)
        f = np.nan_to_num(d["frac"][s])
        x = np.stack([f, d["valid"][s], d["uo"][s], d["vo"][s], d["u10"][s] / 10, d["v10"][s] / 10], 1)
        fut = slice(t + 1, t + HORIZONTE + 1)
        forz = np.stack([d["uo"][fut].mean(0), d["vo"][fut].mean(0),
                         d["u10"][fut].mean(0) / 10, d["v10"][fut].mean(0) / 10])
        y = d["frac"][t + HORIZONTE]
        m = np.isfinite(y) & (d["valid"][t + HORIZONTE] > 0.5) & d["mar"]
        return (torch.from_numpy(x), torch.from_numpy(forz), torch.from_numpy(np.nan_to_num(y)),
                torch.from_numpy(m), torch.tensor([i, t]))


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, ch, k=3):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(cin + ch, 4 * ch, k, padding=k // 2)

    def forward(self, x, hc):
        h, c = hc
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class SargazoConvLSTM(nn.Module):
    def __init__(self, cin=6, ch=48):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(32, 32, 3, padding=1), nn.GELU())
        self.l1 = ConvLSTMCell(32, ch)
        self.l2 = ConvLSTMCell(ch, ch)
        self.cab = nn.Sequential(nn.Conv2d(ch + 4 + 1, 64, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(64, 64, 5, padding=2), nn.GELU(),
                                 nn.Conv2d(64, 1, 1))

    def forward(self, x, forz):
        B, T, C, H, W = x.shape
        h1 = c1 = x.new_zeros(B, self.l1.ch, H, W)
        h2 = c2 = x.new_zeros(B, self.l2.ch, H, W)
        for t in range(T):
            e = self.enc(x[:, t])
            h1, c1 = self.l1(e, (h1, c1))
            h2, c2 = self.l2(h1, (h2, c2))
        hoy = x[:, -1, :1]
        delta = self.cab(torch.cat([h2, forz, hoy], 1))
        return torch.clamp(hoy[:, 0] + delta[:, 0], 0, 1)  # predicción residual sobre la persistencia


def adveccion(d, t, lat):
    """Semi-lagrangiano hacia atrás: valor en x(t+7) = valor de hoy en x - v·7 días."""
    fut = slice(t + 1, t + HORIZONTE + 1)
    u = d["uo"][fut].mean(0) + WINDAGE * d["u10"][fut].mean(0)
    v = d["vo"][fut].mean(0) + WINDAGE * d["v10"][fut].mean(0)
    seg = HORIZONTE * 86400
    dlat = float(np.diff(d["lat"]).mean()); dlon = float(np.diff(d["lon"]).mean())
    di = v * seg / 111e3 / dlat
    dj = u * seg / (111e3 * np.cos(np.deg2rad(lat))[:, None]) / dlon
    H, W = u.shape
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return map_coordinates(np.nan_to_num(d["frac"][t]), [ii - di, jj - dj], order=1, mode="nearest")


def metricas(pred, y, m, umbral=0.05):
    p, t = pred[m], y[m]
    mse = float(((p - t) ** 2).mean())
    pb, tb = p > umbral, t > umbral
    inter, union = (pb & tb).sum(), (pb | tb).sum()
    return mse, float(inter / max(union, 1))


@torch.no_grad()
def evaluar(modelo, ds, dev):
    modelo.eval()
    acc = {k: [] for k in ("modelo", "persistencia", "adveccion")}
    for x, forz, y, m, it in DataLoader(ds, 8):
        pred = modelo(x.to(dev), forz.to(dev)).cpu().numpy()
        for b in range(len(y)):
            i, t = it[b].tolist(); d = ds.t[i]
            yy, mm = y[b].numpy(), m[b].numpy()
            if mm.sum() < 100:
                continue
            acc["modelo"].append(metricas(pred[b], yy, mm))
            acc["persistencia"].append(metricas(np.nan_to_num(d["frac"][t]), yy, mm))
            acc["adveccion"].append(metricas(adveccion(d, t, d["lat"]), yy, mm))
    res = {k: dict(mse=float(np.mean([a[0] for a in v])), iou=float(np.mean([a[1] for a in v])), n=len(v))
           for k, v in acc.items()}
    for k in ("modelo", "adveccion"):
        res[k]["skill_vs_persistencia"] = 1 - res[k]["mse"] / res["persistencia"]["mse"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--lote", type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va, te = Temporadas(range(2019, 2024)), Temporadas([2024]), Temporadas([2025])
    print(f"Muestras: train {len(tr)} | val {len(va)} | test {len(te)} | {dev}", flush=True)
    modelo = SargazoConvLSTM().to(dev)
    print(f"Parámetros: {sum(p.numel() for p in modelo.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(modelo.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epocas)
    escala = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

    salida = RAIZ / "modelos" / "convlstm_sargazo.pt"
    mejor, hist = 1e9, []
    for ep in range(1, a.epocas + 1):
        modelo.train(); t0 = time.time(); acum = 0
        for x, forz, y, m, _ in DataLoader(tr, a.lote, shuffle=True):
            x, forz, y, m = x.to(dev), forz.to(dev), y.to(dev), m.to(dev)
            opt.zero_grad()
            with torch.autocast(dev, enabled=dev == "cuda"):
                pred = modelo(x, forz)
            l = (((pred.float() - y) ** 2)[m]).mean() if m.any() else pred.sum() * 0
            escala.scale(l).backward(); escala.step(opt); escala.update(); acum += l.item()
        sched.step()
        rv = evaluar(modelo, va, dev)
        hist.append(dict(epoca=ep, perdida=acum, val=rv))
        marca = ""
        if rv["modelo"]["mse"] < mejor:
            mejor = rv["modelo"]["mse"]; marca = " *"; torch.save(modelo.state_dict(), salida)
        print(f" época {ep:3d} | val skill {rv['modelo']['skill_vs_persistencia']:+.3f} "
              f"(advección {rv['adveccion']['skill_vs_persistencia']:+.3f}) | IoU {rv['modelo']['iou']:.3f} "
              f"vs pers {rv['persistencia']['iou']:.3f} | {time.time() - t0:.0f}s{marca}", flush=True)

    modelo.load_state_dict(torch.load(salida))
    rt = evaluar(modelo, te, dev)
    print("\nPRUEBA 2025 (horizonte 7 días)")
    for k, v in rt.items():
        extra = f" | skill vs persistencia {v['skill_vs_persistencia']:+.3f}" if "skill_vs_persistencia" in v else ""
        print(f"  {k:<13} MSE {v['mse']:.5f} | IoU {v['iou']:.3f}{extra} (n={v['n']})")
    json.dump(dict(prueba=rt, historial=hist), open(salida.with_name("convlstm_sargazo_metricas.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
