"""Descarga corrientes superficiales (uo, vo) de Copernicus Marine, mes por mes.

Para cada mes prueba, en orden, los datasets disponibles y usa el primero que cubra las fechas:
  1. Análisis/pronóstico  cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m   (desde 2022-06)
  2. Reanálisis           cmems_mod_glo_phy_my_0.083deg_P1D-m          (GLORYS12, histórico)
  3. Reanálisis interino  cmems_mod_glo_phy_myint_0.083deg_P1D-m       (puente entre ambos)

Requisitos:
    pip install copernicusmarine
    copernicusmarine login      # cuenta gratuita en https://data.marine.copernicus.eu/register

Uso:
    python scripts/02_descargar_corrientes.py --inicio 2019-03-01 --fin 2019-09-30
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import copernicusmarine

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN  # noqa: E402

DATASETS = [
    "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
    "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "cmems_mod_glo_phy_myint_0.083deg_P1D-m",
]


def meses(inicio: date, fin: date):
    actual = inicio.replace(day=1)
    while actual <= fin:
        sig = (actual.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield max(actual, inicio), min(sig - timedelta(days=1), fin)
        actual = sig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inicio", required=True, type=date.fromisoformat)
    p.add_argument("--fin", required=True, type=date.fromisoformat)
    a = p.parse_args()

    carpeta = DATA_RAW / "corrientes"
    carpeta.mkdir(parents=True, exist_ok=True)

    for t0, t1 in meses(a.inicio, a.fin):
        nombre = f"corrientes_{t0:%Y%m}.nc"
        if (carpeta / nombre).exists():
            print(f"ya existe {nombre}")
            continue
        for ds in DATASETS:
            try:
                copernicusmarine.subset(
                    dataset_id=ds,
                    variables=["uo", "vo"],
                    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
                    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
                    start_datetime=f"{t0}T00:00:00", end_datetime=f"{t1}T23:59:59",
                    minimum_depth=0, maximum_depth=1,
                    coordinates_selection_method="nearest",
                    output_directory=str(carpeta), output_filename=nombre,
                    disable_progress_bar=True,
                )
                print(f"OK {nombre} ← {ds}", flush=True)
                break
            except Exception as e:  # fuera de rango u otro error: probar el siguiente dataset
                print(f"  {ds} no sirve para {t0:%Y-%m}: {str(e)[:120]}", flush=True)
        else:
            print(f"ERROR: ningún dataset cubre {t0:%Y-%m}", flush=True)


if __name__ == "__main__":
    main()
