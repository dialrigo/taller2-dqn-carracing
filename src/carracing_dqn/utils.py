"""Utilidades: configuración, semillas, métricas de pista y registro en MLflow."""
import random
from pathlib import Path

import numpy as np
import torch
import yaml

RAIZ = Path(__file__).resolve().parents[2]


def cargar_config(ruta: str | Path) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fijar_semillas(semilla: int):
    random.seed(semilla); np.random.seed(semilla); torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def avance_pista(env) -> float:
    """Fracción de baldosas de la pista visitadas en el episodio actual (0 a 1)."""
    u = env.unwrapped
    return u.tile_visited_count / max(len(u.track), 1)


def tipo_fin(env, terminado: bool, truncado: bool, info: dict) -> str:
    if info.get("cortado_sin_progreso"):
        return "sin progreso"
    if terminado:
        return "vuelta completa" if avance_pista(env) >= 0.99 else "fuera del mapa"
    return "límite de tiempo"


def aplanar(d: dict, prefijo: str = "") -> dict:
    out = {}
    for k, v in d.items():
        clave = f"{prefijo}{k}"
        out.update(aplanar(v, clave + ".") if isinstance(v, dict) else {clave: v})
    return out
