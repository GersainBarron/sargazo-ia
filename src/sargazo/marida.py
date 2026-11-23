"""Dataset MARIDA para segmentación binaria de sargazo (Sentinel-2, 11 bandas, 256x256).

Clases originales: 2 = Dense Sargassum, 3 = Sparse Sargassum → positivo (1).
Cualquier otra clase etiquetada (1, 4-15) → negativo (0).
Píxeles sin etiqueta (0) → se ignoran en la pérdida y en las métricas (valor IGNORAR).
"""
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

SARGAZO = (2, 3)
IGNORAR = 255


def ruta_parche(base: Path, pid: str) -> Path:
    # pid ejemplo: "1-12-19_48MYU_0" → patches/S2_1-12-19_48MYU/S2_1-12-19_48MYU_0.tif
    escena = "S2_" + pid.rsplit("_", 1)[0]
    return base / "patches" / escena / f"S2_{pid}.tif"


def leer_split(base: Path, split: str) -> list[str]:
    return (base / "splits" / f"{split}_X.txt").read_text().split()


def cargar(base: Path, pid: str):
    p = ruta_parche(base, pid)
    with rasterio.open(p) as r:
        x = r.read().astype(np.float32)  # (11, 256, 256)
    with rasterio.open(str(p).replace(".tif", "_cl.tif")) as r:
        cl = r.read(1)
    y = np.full(cl.shape, IGNORAR, np.uint8)
    y[cl > 0] = 0
    y[np.isin(cl, SARGAZO)] = 1
    x = np.nan_to_num(x, nan=0.0)
    return x, y


def estadisticas(base: Path, ids: list[str]):
    suma = suma2 = n = 0
    for pid in ids:
        x, _ = cargar(base, pid)
        v = x.reshape(x.shape[0], -1)
        suma = suma + v.sum(1)
        suma2 = suma2 + (v.astype(np.float64) ** 2).sum(1)
        n += v.shape[1]
    media = suma / n
    std = np.sqrt(np.maximum(suma2 / n - media**2, 1e-12))
    return media.astype(np.float32), std.astype(np.float32)


class MaridaSargazo(Dataset):
    def __init__(self, base, ids, media, std, aumentar=False):
        self.base, self.ids, self.aumentar = Path(base), ids, aumentar
        self.media = media[:, None, None]
        self.std = std[:, None, None]
        self.cache = {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        if pid not in self.cache:
            self.cache[pid] = cargar(self.base, pid)
        x, y = self.cache[pid]
        x = (x - self.media) / self.std
        if self.aumentar:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)).long()
