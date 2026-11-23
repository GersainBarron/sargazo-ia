"""ConvLSTM v2: predice la PROBABILIDAD DE PRESENCIA de sargazo a 7 días (no la cantidad).

Cambios respecto a la v1 (scripts/10_entrenar_convlstm.py):
  1. Objetivo binario: celda con cobertura > 5 %  → pérdida focal + Dice (evita el campo "borroso").
  2. Se excluyen las muestras cuyo día actual u objetivo es atípico (scripts/13_dias_atipicos.py).
  3. Paro temprano (paciencia 5 épocas).
  4. Baseline físico serio: trayectorias LAGRANGIANAS de partículas (corriente + windage, paso de 6 h),
     con windage calibrado en validación.
  5. Métrica operativa por TRAMO DE COSTA de Quintana Roo: ¿llegará sargazo a la franja de 20 km
     frente a cada tramo en 7 días? (F1 de la alerta por tramo).
Umbrales de decisión de cada método se eligen en validación (2024) y se reportan en prueba (2025).

Uso:
    python scripts/14_entrenar_convlstm_v2.py --epocas 25
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, map_coordinates
from torch.utils.data import DataLoader

RAIZ = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v1", RAIZ / "scripts" / "10_entrenar_convlstm.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)

H = v1.HORIZONTE
PRESENCIA = 0.05
ATIPICOS = json.load(open(v1.DATOS / "dias_atipicos.json"))
TRAMOS = {  # (lat_min, lat_max) sobre la costa de Quintana Roo (lon −88.3 a −86.5)
    "Cancún–Isla Mujeres": (20.95, 21.65),
    "Puerto Morelos–Playa del Carmen–Cozumel": (20.35, 20.95),
    "Tulum–Sian Ka'an": (19.45, 20.35),
    "Costa Maya–Mahahual–Xcalak": (18.15, 19.45),
}


class TemporadasV2(v1.Temporadas):
    def __init__(self, anios):
        super().__init__(anios)
        malos = {d for v in ATIPICOS.values() for d in v}
        antes = len(self.items)
        self.items = [(i, t) for i, t in self.items
                      if str(self.t[i]["fechas"][t]) not in malos and str(self.t[i]["fechas"][t + H]) not in malos]
        self.excluidas = antes - len(self.items)


class ConvLSTMv2(v1.SargazoConvLSTM):
    def forward(self, x, forz):
        B, T, C, Hh, W = x.shape
        h1 = c1 = x.new_zeros(B, self.l1.ch, Hh, W)
        h2 = c2 = x.new_zeros(B, self.l2.ch, Hh, W)
        for t in range(T):
            e = self.enc(x[:, t])
            h1, c1 = self.l1(e, (h1, c1))
            h2, c2 = self.l2(h1, (h2, c2))
        return self.cab(torch.cat([h2, forz, x[:, -1, :1]], 1))[:, 0]  # logits


def perdida(logits, y, m, alfa=0.75, gamma=2.0):
    z, t = logits[m].float(), (y[m] > PRESENCIA).float()
    if t.numel() == 0:
        return logits.sum() * 0
    p = torch.sigmoid(z)
    bce = F.binary_cross_entropy_with_logits(z, t, reduction="none")
    pt = p * t + (1 - p) * (1 - t)
    focal = ((alfa * t + (1 - alfa) * (1 - t)) * (1 - pt) ** gamma * bce).mean()
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return focal + dice


def lagrangiano(d, t, windage, n_sub=4):
    """Partículas desde cada celda con sargazo hoy, avanzadas 7 días con corriente + windage·viento."""
    f = np.nan_to_num(d["frac"][t])
    ii, jj = np.nonzero((f > 0.01) & d["mar"])
    if len(ii) == 0:
        return np.zeros_like(f)
    rng = np.random.default_rng(0)
    rep = 4
    pi = np.repeat(ii, rep) + rng.uniform(-.5, .5, len(ii) * rep)
    pj = np.repeat(jj, rep) + rng.uniform(-.5, .5, len(jj) * rep)
    w = np.repeat(f[ii, jj], rep) / rep
    vivo = np.ones_like(w, bool)
    dlat = float(np.diff(d["lat"]).mean()); dlon = float(np.diff(d["lon"]).mean())
    coslat = np.cos(np.deg2rad(d["lat"]))
    dt = 86400 / n_sub
    nH, nW = f.shape
    for dia in range(t + 1, t + H + 1):
        u = d["uo"][dia] + windage * d["u10"][dia]
        v = d["vo"][dia] + windage * d["v10"][dia]
        for _ in range(n_sub):
            c = [pi[vivo], pj[vivo]]
            uu = map_coordinates(u, c, order=1, mode="nearest")
            vv = map_coordinates(v, c, order=1, mode="nearest")
            cl = np.interp(pi[vivo], np.arange(nH), coslat)
            pi[vivo] += vv * dt / 111e3 / dlat
            pj[vivo] += uu * dt / (111e3 * cl) / dlon
            fuera = (pi < 0) | (pi > nH - 1) | (pj < 0) | (pj > nW - 1)
            vivo &= ~fuera
            ri, rj = np.clip(np.rint(pi).astype(int), 0, nH - 1), np.clip(np.rint(pj).astype(int), 0, nW - 1)
            vivo &= d["mar"][ri, rj]  # varada en costa / zona excluida → deja de moverse y no cuenta
    out = np.zeros_like(f)
    ri, rj = np.rint(pi[vivo]).astype(int), np.rint(pj[vivo]).astype(int)
    np.add.at(out, (ri, rj), w[vivo])
    return out


def mascaras_tramos(d):
    mar = d["mar"]
    franja = binary_dilation(~mar, iterations=3) & mar  # ~20 km de mar frente a la costa/zona excluida
    lat2 = np.repeat(d["lat"][:, None], len(d["lon"]), 1)
    lon2 = np.repeat(d["lon"][None, :], len(d["lat"]), 0)
    return {n: franja & (lat2 >= a) & (lat2 < b) & (lon2 > -88.3) & (lon2 < -86.5) for n, (a, b) in TRAMOS.items()}


def iou(pb, tb):
    return float((pb & tb).sum() / max((pb | tb).sum(), 1))


@torch.no_grad()
def predecir(modelo, ds, dev, w_lag):
    """Devuelve, por muestra: (y, m, campos dict) con campos = probabilidades/valores de cada método."""
    modelo.eval()
    res = []
    for x, forz, y, m, it in DataLoader(ds, 8):
        p = torch.sigmoid(modelo(x.to(dev), forz.to(dev))).cpu().numpy()
        for b in range(len(y)):
            i, t = it[b].tolist(); d = ds.t[i]
            if m[b].sum() < 100:
                continue
            res.append(dict(d=d, y=y[b].numpy(), m=m[b].numpy(), campos=dict(
                modelo=p[b], persistencia=np.nan_to_num(d["frac"][t]), lagrangiano=lagrangiano(d, t, w_lag))))
    return res


def evaluar(res, umbrales):
    out = {}
    for k, u in umbrales.items():
        ious, tp, fp, fn = [], 0, 0, 0
        for r in res:
            tb = (r["y"] > PRESENCIA) & r["m"]; pb = (r["campos"][k] > u) & r["m"]
            ious.append(iou(pb, tb))
            for mt in mascaras_tramos(r["d"]).values():
                real, alerta = (tb & mt).sum() >= 2, (pb & mt).sum() >= 2
                tp += real and alerta; fp += alerta and not real; fn += real and not alerta
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        out[k] = dict(iou=float(np.mean(ious)), f1_tramos=float(f1), tp=int(tp), fp=int(fp), fn=int(fn), umbral=float(u))
    return out


def elegir_umbrales(res):
    cand = {"modelo": np.arange(0.2, 0.91, 0.05), "persistencia": [0.02, 0.05, 0.1, 0.15, 0.2],
            "lagrangiano": [0.01, 0.02, 0.05, 0.1, 0.15]}
    return {k: max(c, key=lambda u: evaluar(res, {k: u})[k]["iou"]) for k, c in cand.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=25)
    ap.add_argument("--paciencia", type=int, default=5)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va, te = TemporadasV2(range(2019, 2024)), TemporadasV2([2024]), TemporadasV2([2025])
    print(f"Muestras: train {len(tr)} | val {len(va)} | test {len(te)} "
          f"(excluidas por días atípicos: {tr.excluidas + va.excluidas + te.excluidas})", flush=True)

    # Windage del baseline lagrangiano, calibrado en validación (submuestra)
    sub = [va[k] for k in range(0, len(va), 4)]
    mejor_w, mejor_iou = 0.0, -1
    for w in (0.0, 0.01, 0.02, 0.03):
        ious = []
        for (x, forz, y, m, it) in sub:
            i, t = it.tolist(); d = va.t[i]
            tb = (y.numpy() > PRESENCIA) & m.numpy()
            ious.append(iou((lagrangiano(d, t, w) > 0.02) & m.numpy(), tb))
        print(f"  lagrangiano windage {w:.2f}: IoU val {np.mean(ious):.3f}", flush=True)
        if np.mean(ious) > mejor_iou:
            mejor_w, mejor_iou = w, np.mean(ious)

    modelo = ConvLSTMv2().to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), 1e-3, weight_decay=1e-4)
    escala = torch.amp.GradScaler(enabled=dev == "cuda")
    salida = RAIZ / "modelos" / "convlstm_v2_sargazo.pt"
    mejor, sin_mejora, hist = -1, 0, []
    for ep in range(1, a.epocas + 1):
        modelo.train(); t0 = time.time(); acum = 0
        for x, forz, y, m, _ in DataLoader(tr, 4, shuffle=True):
            x, forz, y, m = x.to(dev), forz.to(dev), y.to(dev), m.to(dev)
            opt.zero_grad()
            with torch.autocast(dev, enabled=dev == "cuda"):
                logits = modelo(x, forz)
            l = perdida(logits, y, m)
            escala.scale(l).backward(); escala.step(opt); escala.update(); acum += l.item()
        # validación rápida con umbral 0.5
        modelo.eval(); ious = []
        with torch.no_grad():
            for x, forz, y, m, _ in DataLoader(va, 8):
                p = torch.sigmoid(modelo(x.to(dev), forz.to(dev))).cpu().numpy() > 0.5
                for b in range(len(y)):
                    mm = m[b].numpy()
                    if mm.sum() >= 100:
                        ious.append(iou(p[b] & mm, (y[b].numpy() > PRESENCIA) & mm))
        v = float(np.mean(ious)); hist.append(dict(epoca=ep, perdida=acum / len(tr), val_iou=v))
        marca = ""
        if v > mejor:
            mejor, sin_mejora, marca = v, 0, " *"; torch.save(modelo.state_dict(), salida)
        else:
            sin_mejora += 1
        print(f" época {ep:3d} | pérdida {acum / len(tr):.4f} | val IoU {v:.3f} | {time.time() - t0:.0f}s{marca}", flush=True)
        if sin_mejora >= a.paciencia:
            print(" paro temprano", flush=True); break

    modelo.load_state_dict(torch.load(salida))
    print("\nEligiendo umbrales en validación 2024…", flush=True)
    umb = elegir_umbrales(predecir(modelo, va, dev, mejor_w))
    print("Umbrales:", {k: round(float(u), 2) for k, u in umb.items()}, f"| windage lagrangiano {mejor_w}", flush=True)
    rt = evaluar(predecir(modelo, te, dev, mejor_w), umb)
    print("\nPRUEBA 2025 · presencia de sargazo a 7 días")
    print(f"  {'método':<14}{'IoU':>7}{'F1 tramos':>11}   (TP/FP/FN alertas por tramo)")
    for k, r in rt.items():
        print(f"  {k:<14}{r['iou']:>7.3f}{r['f1_tramos']:>11.3f}   ({r['tp']}/{r['fp']}/{r['fn']})")
    json.dump(dict(prueba=rt, umbrales={k: float(u) for k, u in umb.items()}, windage=mejor_w, historial=hist),
              open(salida.with_name("convlstm_v2_sargazo_metricas.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
