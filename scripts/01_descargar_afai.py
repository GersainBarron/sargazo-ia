"""Descarga el índice AFAI (USF/NOAA) desde ERDDAP, mes por mes, en NetCDF.

No requiere cuenta.

Uso:
    python scripts/01_descargar_afai.py --inicio 2025-03-01 --fin 2025-08-31 [--producto 7D|1D]
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN  # noqa: E402

SERVIDOR = "https://cwcgom.aoml.noaa.gov/erddap/griddap"
PRODUCTOS = {
    "1D": "noaa_aoml_atlantic_oceanwatch_AFAI_1D",
    "7D": "noaa_aoml_atlantic_oceanwatch_AFAI_7D",
}


def meses(inicio: date, fin: date):
    actual = inicio.replace(day=1)
    while actual <= fin:
        sig = (actual.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield max(actual, inicio), min(sig - timedelta(days=1), fin)
        actual = sig


def url(dataset: str, t0: date, t1: date, lat_desc: bool) -> str:
    lat = f"[({LAT_MAX}):1:({LAT_MIN})]" if lat_desc else f"[({LAT_MIN}):1:({LAT_MAX})]"
    return (
        f"{SERVIDOR}/{dataset}.nc?AFAI"
        f"[({t0}T00:00:00Z):1:({t1}T23:59:59Z)]"
        f"{lat}[({LON_MIN}):1:({LON_MAX})]"
    )


def descargar(dataset: str, t0: date, t1: date, destino: Path) -> None:
    if destino.exists():
        print(f"  ya existe {destino.name}, se omite")
        return
    # ERDDAP exige el orden de latitud del dataset; probamos ascendente y luego descendente.
    for lat_desc in (False, True):
        r = requests.get(url(dataset, t0, t1, lat_desc), stream=True, timeout=600)
        if r.status_code == 200:
            tmp = destino.with_suffix(".part")
            with open(tmp, "wb") as f:
                for bloque in r.iter_content(1 << 20):
                    f.write(bloque)
            tmp.rename(destino)
            print(f"  OK {destino.name} ({destino.stat().st_size / 1e6:.1f} MB)")
            return
        ultimo = f"HTTP {r.status_code}: {r.text[:300]}"
    print(f"  ERROR {t0}–{t1}: {ultimo}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inicio", required=True, type=date.fromisoformat)
    p.add_argument("--fin", required=True, type=date.fromisoformat)
    p.add_argument("--producto", default="7D", choices=PRODUCTOS)
    a = p.parse_args()

    dataset = PRODUCTOS[a.producto]
    carpeta = DATA_RAW / "afai" / a.producto
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"Dataset {dataset} → {carpeta}")
    for t0, t1 in meses(a.inicio, a.fin):
        descargar(dataset, t0, t1, carpeta / f"afai_{a.producto}_{t0:%Y%m}.nc")


if __name__ == "__main__":
    main()
