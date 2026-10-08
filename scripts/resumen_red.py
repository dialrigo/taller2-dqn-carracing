"""Imprime la arquitectura de la red Q con las dimensiones reales de cada capa.

    uv run python scripts/resumen_red.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carracing_dqn.model import RedQ, resumen_capas  # noqa: E402

red = RedQ(canales=4, n_acciones=5)
print(f"{'Capa':<10} {'Salida (C, H, W)':<20} {'Parámetros':>12}")
total = 0
for nombre, forma, n in resumen_capas(red):
    print(f"{nombre:<10} {str(forma):<20} {n:>12,}"); total += n
print(f"{'Total':<31} {total:>12,}")
