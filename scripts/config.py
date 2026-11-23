"""Configuración común del proyecto Sargazo IA."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw"

# Caribe mexicano + zona de aproximación
LAT_MIN, LAT_MAX = 14.0, 24.0
LON_MIN, LON_MAX = -90.0, -80.0
