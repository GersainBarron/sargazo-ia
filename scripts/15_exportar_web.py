"""Exporta los datos de la temporada 2025 y las predicciones de la ConvLSTM v2 para el simulador web (Three.js).

Salida en web/simulador/data/:
  obs.png       atlas de cobertura observada por día (R = cobertura/0.5, 167×167 por mosaico)
  pred.png      atlas de probabilidad predicha para cada día objetivo (pronóstico emitido 7 días antes)
  corr.png      atlas de corrientes (R = u, G = v; 128 = 0, ±1.5 m/s), 84×84 por mosaico
  tierra.png    máscara de tierra a 667×667 (R) + distancia a la costa (G, 0–255 ≈ 0–40 km)
  meta.json     fechas, límites, tamaños de atlas, tramos de costa, métricas por día y globales

Uso:
    python scripts/15_exportar_web.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "web" / "simulador" / "data"
SALIDA.mkdir(parents=True, exist_ok=True)


def cargar_modulo(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / "scripts" / archivo)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


v1 = cargar_modulo("v1", "10_entrenar_convlstm.py")
v2 = cargar_modulo("v2", "14_entrenar_convlstm_v2.py")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def atlas(capas, columnas=16):
    """capas: (N, H, W, C) uint8 → imagen (filas*H, columnas*W, C)."""
    n, h, w, c = capas.shape
    filas = int(np.ceil(n / columnas))
    img = np.zeros((filas * h, columnas * w, c), np.uint8)
    for i in range(n):
        f, k = divmod(i, columnas)
        img[f * h:(f + 1) * h, k * w:(k + 1) * w] = capas[i]
    return img, filas, columnas


def main():
    ds = v1.Temporadas([2025])
    d = ds.t[0]
    fechas = [str(f) for f in d["fechas"]]
    T, H, W = d["frac"].shape
    mar = d["mar"]
    # Las filas de la imagen van de norte a sur (latitud descendente)
    voltear = lambda a: a[..., ::-1, :] if a.ndim == 3 else a[::-1, :]

    # ---------- observado
    frac = np.nan_to_num(d["frac"])
    obs = np.clip(frac / 0.5, 0, 1) * 255
    obs = voltear(np.where(mar[None], obs, 0)).astype(np.uint8)[..., None]
    img, filas, cols = atlas(obs)
    Image.fromarray(img[..., 0], "L").save(SALIDA / "obs.png", optimize=True)

    # ---------- corrientes y viento
    def cod(x, rango):
        return (np.clip(x / rango, -1, 1) * 127 + 128).astype(np.uint8)
    # RGB (sin alfa: el navegador premultiplica el alfa y alteraría los valores)
    corr = np.stack([cod(d["uo"], 1.5), cod(d["vo"], 1.5), np.full_like(cod(d["uo"], 1.5), 128)], -1)
    corr = corr[:, ::-1][:, ::2, ::2]  # norte arriba; mitad de resolución (campo suave)
    img_c, _, _ = atlas(corr)
    Image.fromarray(img_c, "RGB").save(SALIDA / "corr.png", optimize=True)

    # ---------- predicción ConvLSTM v2 (por día objetivo) + métricas por día
    modelo = v2.ConvLSTMv2().to(dev)
    modelo.load_state_dict(torch.load(RAIZ / "modelos" / "convlstm_v2_sargazo.pt", map_location=dev))
    modelo.eval()
    metr_v2 = json.load(open(RAIZ / "modelos" / "convlstm_v2_sargazo_metricas.json", encoding="utf8"))
    umbral = 0.75
    pred = np.zeros((T, H, W), np.float32)
    tiene_pred = [False] * T
    tramos_m = v2.mascaras_tramos(d)
    por_dia = {}
    atipicos = {x for v in v2.ATIPICOS.values() for x in v}
    with torch.no_grad():
        for k in range(len(ds)):
            _, t = ds.items[k]
            x, forz, y, m, _ = ds[k]
            p = torch.sigmoid(modelo(x[None].to(dev), forz[None].to(dev)))[0].cpu().numpy()
            obj = t + v1.HORIZONTE
            pred[obj] = p; tiene_pred[obj] = True
            mm = m.numpy(); tb = (y.numpy() > v2.PRESENCIA) & mm
            iou_m = v2.iou((p > umbral) & mm, tb)
            iou_p = v2.iou((frac[t] > 0.02) & mm, tb)
            tr = {}
            for nombre, mt in tramos_m.items():
                tr[nombre] = dict(
                    prob=round(float(p[mt].mean()), 3),
                    obs=round(float(frac[obj][mt].mean()), 4),
                    obs_presencia=bool((tb & mt).sum() >= 2),
                )
            por_dia[fechas[obj]] = dict(emitido=fechas[t], iou_modelo=round(iou_m, 3), iou_persistencia=round(iou_p, 3),
                                        atipico=fechas[obj] in atipicos or fechas[t] in atipicos, tramos=tr)
    pr = voltear(np.where(mar[None], pred, 0) * 255).astype(np.uint8)[..., None]
    img_p, _, _ = atlas(pr)
    Image.fromarray(img_p[..., 0], "L").save(SALIDA / "pred.png", optimize=True)

    # ---------- tierra fina (667×667) y distancia a la costa
    f0 = sorted((RAIZ / "data" / "raw" / "afai" / "7D").glob("afai_7D_2025*.nc"))
    a = xr.concat([xr.open_dataset(f)["AFAI"] for f in f0], dim="time")
    tierra = a.isnull().all("time").values
    # Limpiar píxeles sueltos (nubes persistentes) y suavizar
    tierra = gaussian_filter(tierra.astype(np.float32), 1.0) > 0.5
    dist = distance_transform_edt(~tierra) * 1.6  # km
    img_t = np.stack([tierra * 255, np.clip(dist / 40 * 255, 0, 255), np.zeros_like(dist)], -1).astype(np.uint8)
    Image.fromarray(img_t[::-1], "RGB").save(SALIDA / "tierra.png", optimize=True)

    # ---------- tramos: posición (para dibujarlos) y metadatos
    tramos = []
    for nombre, (la, lb) in v2.TRAMOS.items():
        mt = tramos_m[nombre]
        ii, jj = np.nonzero(mt)
        tramos.append(dict(nombre=nombre, lat=[la, lb],
                           centro=[float(d["lat"][ii].mean()), float(d["lon"][jj].mean())]))

    unet = json.load(open(RAIZ / "modelos" / "unet_marida_resnet34_metricas.json", encoding="utf8"))
    meta = dict(
        autor="Gersain Barrón", temporada=2025, fechas=fechas, horizonte_dias=v1.HORIZONTE,
        lat=[float(d["lat"][0]), float(d["lat"][-1])], lon=[float(d["lon"][0]), float(d["lon"][-1])],
        rejilla=[H, W], rejilla_corr=list(corr.shape[1:3]), atlas=dict(columnas=cols, filas=filas, n=T), tierra_px=list(tierra.shape),
        escala=dict(obs=0.5, corriente=1.5, viento=15),
        tiene_pred=tiene_pred, umbral_modelo=umbral, por_dia=por_dia, tramos=tramos,
        metricas=dict(
            prediccion_prueba_2025=metr_v2.get("prueba_2025_presencia_7d") or metr_v2.get("prueba"),
            unet_prueba=dict(f1=round(unet["prueba"]["f1"], 3), iou=round(unet["prueba"]["iou"], 3),
                             precision=round(unet["prueba"]["precision"], 3), recall=round(unet["prueba"]["recall"], 3)),
        ),
        fuentes=["NOAA AOML / USF AFAI 7D (ERDDAP)", "Copernicus Marine GLOBAL_ANALYSISFORECAST_PHY_001_024 y GLORYS12",
                 "ERA5 (Copernicus Climate Data Store)", "MARIDA (Kikaki et al., 2022)"],
    )
    json.dump(meta, open(SALIDA / "meta.json", "w", encoding="utf8"), ensure_ascii=False)
    tam = {f.name: f"{f.stat().st_size / 1e6:.2f} MB" for f in SALIDA.iterdir()}
    print(f"{T} días, predicciones para {sum(tiene_pred)} días, atlas {filas}×{cols}")
    print(tam)


if __name__ == "__main__":
    main()
