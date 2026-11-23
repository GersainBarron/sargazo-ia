"""Descarga MARIDA (Marine Debris Archive, Sentinel-2, incluye clase Sargassum) desde Zenodo.

Cita: Kikaki K. et al. (2022) PLoS ONE 17(1): e0262247. https://zenodo.org/records/5151941

Uso:
    python scripts/04_descargar_marida.py
"""
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW  # noqa: E402

RECORD = "5151941"


def main():
    carpeta = DATA_RAW / "marida"
    carpeta.mkdir(parents=True, exist_ok=True)
    meta = requests.get(f"https://zenodo.org/api/records/{RECORD}", timeout=60).json()
    for f in meta["files"]:
        nombre = f["key"]
        destino = carpeta / nombre
        if destino.exists():
            print(f"ya existe {nombre}")
            continue
        print(f"Descargando {nombre} ({f['size'] / 1e6:.0f} MB)…")
        with requests.get(f["links"]["self"], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(destino, "wb") as out:
                for bloque in r.iter_content(1 << 20):
                    out.write(bloque)
        if nombre.endswith(".zip"):
            with zipfile.ZipFile(destino) as z:
                z.extractall(carpeta)
            print(f"  descomprimido en {carpeta}")


if __name__ == "__main__":
    main()
