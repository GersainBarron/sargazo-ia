"""Entrena una U-Net para detectar sargazo en Sentinel-2 con MARIDA.

Uso:
    python scripts/07_entrenar_unet.py --epocas 40 --encoder resnet34
Salida:
    modelos/unet_marida_<encoder>.pt  y  modelos/unet_marida_<encoder>_metricas.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from sargazo.marida import IGNORAR, MaridaSargazo, cargar, estadisticas, leer_split  # noqa: E402

BASE = RAIZ / "data" / "raw" / "marida"


def perdida(logits, y, pos_weight):
    m = y != IGNORAR
    if m.sum() == 0:
        return logits.sum() * 0
    z, t = logits[:, 0][m], y[m].float()
    bce = F.binary_cross_entropy_with_logits(z, t, pos_weight=pos_weight)
    p = torch.sigmoid(z)
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return bce + dice


@torch.no_grad()
def evaluar(modelo, loader, dev, umbral=0.5):
    modelo.eval()
    tp = fp = fn = tn = 0
    for x, y in loader:
        p = torch.sigmoid(modelo(x.to(dev)))[:, 0].cpu() > umbral
        m = y != IGNORAR
        t = y == 1
        tp += int((p & t & m).sum()); fp += int((p & ~t & m).sum())
        fn += int((~p & t & m).sum()); tn += int((~p & ~t & m).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9); iou = tp / max(tp + fp + fn, 1)
    return dict(precision=prec, recall=rec, f1=f1, iou=iou, tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=40)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ids = {s: leer_split(BASE, s) for s in ("train", "val", "test")}
    media, std = estadisticas(BASE, ids["train"])
    tiene = [bool((cargar(BASE, i)[1] == 1).any()) for i in ids["train"]]
    print(f"Dispositivo: {dev} | train {len(ids['train'])} ({sum(tiene)} con sargazo) | "
          f"val {len(ids['val'])} | test {len(ids['test'])}")

    # Sobremuestrear parches con sargazo (clase rara)
    pesos = [5.0 if t else 1.0 for t in tiene]
    tr = MaridaSargazo(BASE, ids["train"], media, std, aumentar=True)
    dl_tr = DataLoader(tr, a.lote, sampler=WeightedRandomSampler(pesos, len(pesos)), num_workers=0)
    dl_va = DataLoader(MaridaSargazo(BASE, ids["val"], media, std), a.lote, num_workers=0)
    dl_te = DataLoader(MaridaSargazo(BASE, ids["test"], media, std), a.lote, num_workers=0)

    modelo = smp.Unet(a.encoder, encoder_weights="imagenet", in_channels=len(media), classes=1).to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epocas)
    pos_weight = torch.tensor(10.0, device=dev)

    salida = RAIZ / "modelos" / f"unet_marida_{a.encoder}.pt"
    mejor, historial = -1, []
    for ep in range(1, a.epocas + 1):
        modelo.train(); t0 = time.time(); acum = 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            l = perdida(modelo(x), y, pos_weight)
            l.backward(); opt.step(); acum += l.item()
        sched.step()
        mv = evaluar(modelo, dl_va, dev)
        historial.append(dict(epoca=ep, perdida=acum / len(dl_tr), **{f"val_{k}": v for k, v in mv.items()}))
        marca = ""
        if mv["f1"] > mejor:
            mejor = mv["f1"]; marca = " *"
            torch.save(dict(modelo=modelo.state_dict(), media=media, std=std, encoder=a.encoder), salida)
        print(f" época {ep:3d} | pérdida {acum / len(dl_tr):.3f} | val F1 {mv['f1']:.3f} "
              f"IoU {mv['iou']:.3f} P {mv['precision']:.3f} R {mv['recall']:.3f} | {time.time() - t0:.0f}s{marca}",
              flush=True)

    ck = torch.load(salida, weights_only=False)
    modelo.load_state_dict(ck["modelo"])
    # Elegir umbral en validación y reportar en prueba
    mejores = max(((u, evaluar(modelo, dl_va, dev, u)["f1"]) for u in np.arange(0.3, 0.91, 0.05)), key=lambda t: t[1])
    mt = evaluar(modelo, dl_te, dev, mejores[0])
    print(f"\nUmbral elegido en val: {mejores[0]:.2f} (F1 val {mejores[1]:.3f})")
    print(f"PRUEBA → F1 {mt['f1']:.3f} | IoU {mt['iou']:.3f} | precisión {mt['precision']:.3f} | recall {mt['recall']:.3f}")
    json.dump(dict(encoder=a.encoder, umbral=float(mejores[0]), prueba=mt, historial=historial),
              open(salida.with_name(salida.stem + "_metricas.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
