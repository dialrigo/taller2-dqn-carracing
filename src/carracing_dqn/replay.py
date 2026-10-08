"""Memoria de experiencia circular en uint8.

Cada transición guarda el estado apilado (4x84x84) y solo el cuadro NUEVO del siguiente estado: el
siguiente estado es los 3 últimos cuadros de s más ese cuadro. Así una transición ocupa ~35 KB en lugar
de ~56 KB, y 80 000 transiciones caben en ~2,8 GB.

Si se indica `carpeta`, los arreglos viven en disco (numpy memmap): la memoria sobrevive a un corte del
proceso y el entrenamiento puede reanudarse sin volver a llenarla.
"""
from pathlib import Path

import numpy as np
import torch


class MemoriaExperiencia:
    def __init__(self, capacidad: int, forma=(4, 84, 84), semilla: int = 0, carpeta=None):
        self.capacidad = capacidad
        specs = {"s": ((capacidad, *forma), np.uint8), "nuevo": ((capacidad, *forma[1:]), np.uint8),
                 "a": ((capacidad,), np.int64), "r": ((capacidad,), np.float32),
                 "fin": ((capacidad,), np.float32)}               # fin = 1 solo si el episodio TERMINÓ (no si se truncó)
        for nombre, (shape, dtype) in specs.items():
            if carpeta is None:
                arr = np.zeros(shape, dtype=dtype)
            else:
                ruta = Path(carpeta) / f"{nombre}.npy"; Path(carpeta).mkdir(parents=True, exist_ok=True)
                modo = "r+" if ruta.exists() else "w+"
                arr = np.lib.format.open_memmap(ruta, mode=modo, dtype=dtype, shape=shape)
            setattr(self, nombre, arr)
        self.pos = self.n = 0
        self.rng = np.random.default_rng(semilla)

    def agregar(self, s, a, r, s2, terminado):
        i = self.pos
        self.s[i], self.nuevo[i], self.a[i], self.r[i], self.fin[i] = s, s2[-1], a, r, float(terminado)
        self.pos = (self.pos + 1) % self.capacidad
        self.n = min(self.n + 1, self.capacidad)

    def muestrear(self, batch: int):
        idx = self.rng.integers(0, self.n, size=batch)
        s = self.s[idx]
        s2 = np.concatenate([s[:, 1:], self.nuevo[idx][:, None]], axis=1)
        return (torch.from_numpy(s), torch.from_numpy(self.a[idx]), torch.from_numpy(self.r[idx]),
                torch.from_numpy(s2), torch.from_numpy(self.fin[idx]))

    def guardar_estado(self) -> dict:
        for nombre in ["s", "nuevo", "a", "r", "fin"]:
            arr = getattr(self, nombre)
            if hasattr(arr, "flush"):
                arr.flush()
        return {"pos": self.pos, "n": self.n}

    def restaurar_estado(self, estado: dict):
        self.pos, self.n = estado["pos"], estado["n"]

    def __len__(self):
        return self.n
