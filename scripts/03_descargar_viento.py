"""Descarga viento a 10 m (u10, v10) de ERA5, mes por mes.

Requisitos:
    pip install cdsapi
    Cuenta gratuita en https://cds.climate.copernicus.eu, aceptar la licencia de ERA5
    y crear ~/.cdsapirc con:
        url: https://cds.climate.copernicus.eu/api
        key: <TU_API_KEY>

Uso:
    python scripts/03_descargar_viento.py --inicio 2025-03-01 --fin 2025-08-31
"""
import argparse
import calendar
import sys
from datetime import date
from pathlib import Path

import cdsapi

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inicio", required=True, type=date.fromisoformat)
    p.add_argument("--fin", required=True, type=date.fromisoformat)
    a = p.parse_args()

    carpeta = DATA_RAW / "viento"
    carpeta.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()

    anio, mes = a.inicio.year, a.inicio.month
    while (anio, mes) <= (a.fin.year, a.fin.month):
        destino = carpeta / f"era5_viento_{anio}{mes:02d}.nc"
        if not destino.exists():
            dias = calendar.monthrange(anio, mes)[1]
            print(f"Pidiendo {anio}-{mes:02d}…")
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                    "year": [str(anio)],
                    "month": [f"{mes:02d}"],
                    "day": [f"{d:02d}" for d in range(1, dias + 1)],
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    "area": [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],  # N, W, S, E
                    "data_format": "netcdf",
                },
                str(destino),
            )
        mes += 1
        if mes == 13:
            anio, mes = anio + 1, 1


if __name__ == "__main__":
    main()
