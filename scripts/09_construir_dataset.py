"""Construye el dataset espaciotemporal para la ConvLSTM (una temporada por archivo .npz).

Por día y por celda de la rejilla gruesa (AFAI 0.015° agregada ×4 ≈ 0.06°, ~6.7 km):
  frac   fracción de píxeles finos con AFAI > umbral (cobertura de sargazo, 0–1; NaN si no hay datos)
  valid  fracción de píxeles finos con dato válido (nubes / brillo solar → 0)
  uo, vo corriente superficial (m/s), Copernicus Marine, interpolada
  u10, v10 viento a 10 m (m/s), ERA5, promedio diario interpolado

Máscara: tierra + franja costera (3 px finos) + rasgos estáticos (>50 % de días sobre el umbral
en TODAS las temporadas) se excluyen. Es un modelo de sargazo "mar adentro"; la costa la cubre la U-Net S2.

Uso:
    python scripts/09_construir_dataset.py --anios 2019 2020 2021 2022 2023 2024 2025
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW, RAIZ  # noqa: E402

UMBRAL = 0.0005
FACTOR = 4
SALIDA = RAIZ / "data" / "processed" / "convlstm"


def cargar_afai(anio):
    fs = sorted((DATA_RAW / "afai" / "7D").glob(f"afai_7D_{anio}*.nc"))
    a = xr.concat([xr.open_dataset(f)["AFAI"] for f in fs], dim="time").load()
    a["time"] = a.time.dt.floor("D")
    return a


def mascara_global(anios):
    """Tierra, franja costera y rasgos estáticos en la rejilla fina, usando todas las temporadas."""
    sobre = cnt = None
    tierra = None
    for y in anios:
        a = cargar_afai(y)
        s = (a > UMBRAL).sum("time").values
        c = a.notnull().sum("time").values
        sobre = s if sobre is None else sobre + s
        cnt = c if cnt is None else cnt + c
    tierra = cnt == 0
    franja = binary_dilation(tierra, iterations=3) & ~tierra
    estatico = (sobre / np.maximum(cnt, 1) > 0.5) & ~tierra
    return tierra | franja | estatico, tierra


def gruesa(arr2d_bool_o_float, h, w):
    a = arr2d_bool_o_float[: h * FACTOR, : w * FACTOR].astype(np.float32)
    return a.reshape(h, FACTOR, w, FACTOR).mean((1, 3))


def cargar_forzamiento(carpeta, patron, anio, rejilla):
    """Abre cada archivo por separado, lo interpola a la rejilla gruesa y luego une en el tiempo.
    (Los datasets de Copernicus anfc / my / myint tienen rejillas ligeramente distintas.)"""
    fs = sorted((DATA_RAW / carpeta).glob(patron.format(anio=anio)))
    if not fs:
        return None
    partes = []
    for f in fs:
        ds = xr.open_dataset(f)
        if "valid_time" in ds.dims:
            ds = ds.rename(valid_time="time")
        for d in ("depth", "number", "expver"):
            if d in ds.dims:
                ds = ds.isel({d: 0})
        ds = ds.drop_vars([v for v in ("depth", "number", "expver") if v in ds.coords], errors="ignore")
        ds = ds.sortby("latitude").resample(time="1D").mean().load()
        partes.append(ds.interp(**rejilla))
    # "nearest" en Copernicus puede incluir el 1.º del mes siguiente → quitar fechas repetidas
    return xr.concat(partes, dim="time").sortby("time").drop_duplicates("time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anios", nargs="+", type=int, default=list(range(2019, 2026)))
    a = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)

    print("Calculando máscara global (todas las temporadas disponibles)…", flush=True)
    todos = sorted({int(f.name[8:12]) for f in (DATA_RAW / "afai" / "7D").glob("afai_7D_*.nc")})
    excluir, tierra = mascara_global(todos)

    for y in a.anios:
        afai = cargar_afai(y)
        nlat, nlon = afai.sizes["latitude"] // FACTOR, afai.sizes["longitude"] // FACTOR
        lat = afai.latitude.values[: nlat * FACTOR].reshape(nlat, FACTOR).mean(1)
        lon = afai.longitude.values[: nlon * FACTOR].reshape(nlon, FACTOR).mean(1)

        v = afai.values  # (T, 667, 667)
        ok = np.isfinite(v) & ~excluir[None]
        pos = (v > UMBRAL) & ok
        T = v.shape[0]
        H, W = nlat, nlon
        recorte = (slice(None), slice(0, H * FACTOR), slice(0, W * FACTOR))
        okc = ok[recorte].reshape(T, H, FACTOR, W, FACTOR).sum((2, 4)).astype(np.float32)
        posc = pos[recorte].reshape(T, H, FACTOR, W, FACTOR).sum((2, 4)).astype(np.float32)
        valid = okc / FACTOR**2
        frac = np.where(okc > 0, posc / np.maximum(okc, 1), np.nan).astype(np.float32)
        mar = gruesa(~excluir, H, W) > 0.5

        fechas = pd.DatetimeIndex(afai.time.values)
        rejilla = dict(latitude=xr.DataArray(lat, dims="latitude"), longitude=xr.DataArray(lon, dims="longitude"))

        cur = cargar_forzamiento("corrientes", "corrientes_{anio}*.nc", y, rejilla)
        vie = cargar_forzamiento("viento", "era5_viento_{anio}*.nc", y, rejilla)
        if cur is None or vie is None:
            print(f"{y}: faltan corrientes o viento, se omite", flush=True)
            continue
        cur = cur.reindex(time=fechas, method="nearest", tolerance=pd.Timedelta("1D"))
        vie = vie.reindex(time=fechas, method="nearest", tolerance=pd.Timedelta("1D"))

        def arr(ds, n):
            return np.nan_to_num(ds[n].values.astype(np.float32), nan=0.0)

        np.savez_compressed(
            SALIDA / f"temporada_{y}.npz",
            frac=frac.astype(np.float16), valid=valid.astype(np.float16),
            uo=arr(cur, "uo"), vo=arr(cur, "vo"), u10=arr(vie, "u10"), v10=arr(vie, "v10"),
            fechas=fechas.strftime("%Y-%m-%d").values, lat=lat, lon=lon, mar=mar,
        )
        cobertura = np.nanmean(frac[:, mar])
        print(f"{y}: {T} días, rejilla {H}×{W}, cobertura media {cobertura * 100:.2f} %, "
              f"válidos {valid[:, mar].mean() * 100:.0f} %", flush=True)


if __name__ == "__main__":
    main()
