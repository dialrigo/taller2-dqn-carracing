"""Red Q convolucional (arquitectura de Mnih et al., 2015) para entradas de 4 x 84 x 84."""
import torch
import torch.nn as nn


class RedQ(nn.Module):
    """
    Entrada (B, 4, 84, 84) uint8  ->  /255  ->
    Conv 4->32,  kernel 8x8, stride 4  -> (32, 20, 20)  ReLU
    Conv 32->64, kernel 4x4, stride 2  -> (64, 9, 9)    ReLU
    Conv 64->64, kernel 3x3, stride 1  -> (64, 7, 7)    ReLU
    Aplanar 3136 -> Densa 512 ReLU -> Densa n_acciones (lineal): Q(s, a) para cada acción
    """

    def __init__(self, canales: int = 4, n_acciones: int = 5):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(canales, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.cabeza = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, n_acciones))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.convs(x.float() / 255.0))   # la normalización ocurre dentro de la red


def resumen_capas(red: RedQ, entrada=(1, 4, 84, 84)):
    """Tabla de dimensiones y parámetros por capa, calculada con una pasada real."""
    filas, x = [], torch.zeros(entrada)
    for capa in list(red.convs) + list(red.cabeza):
        x = capa(x)
        n = sum(p.numel() for p in capa.parameters())
        if n or isinstance(capa, nn.Flatten):
            filas.append((capa.__class__.__name__, tuple(x.shape[1:]), n))
    return filas
