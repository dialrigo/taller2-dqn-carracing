"""
Diagnóstico del colapso del modelo final (README, sección 7).   uv run python scripts/diagnostico_colapso.py

1. Valores Q de las 5 acciones en el estado inicial (4 cuadros idénticos, auto detenido) para el mejor
   modelo y el modelo final, en las 50 pistas de evaluación (semillas 1000-1049).
2. Si la imagen deja de cambiar cuando el modelo final elige girar con el auto detenido.
3. Desempeño de ambos modelos cuando se fuerzan 5 decisiones iniciales de «acelerar» (arranque forzado),
   para separar la trampa del estado inicial de la calidad de la conducción.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from carracing_dqn.envs import NOMBRES_ACCIONES, crear_ambiente  # noqa: E402
from carracing_dqn.model import RedQ  # noqa: E402
from carracing_dqn.utils import cargar_config  # noqa: E402

cfg = cargar_config(RAIZ / "configs" / "dqn_carracing.yaml")
env = crear_ambiente(cfg["ambiente"], entrenamiento=False)
SEMILLAS = range(cfg["evaluacion"]["semilla_base"], cfg["evaluacion"]["semilla_base"] + cfg["evaluacion"]["episodios"])


def cargar(nombre):
    ck = torch.load(RAIZ / "models" / f"dqn_carracing_{nombre}.pt", map_location="cpu", weights_only=False)
    red = RedQ(); red.load_state_dict(ck["red"]); red.eval()
    return red


def q(red, obs):
    with torch.no_grad():
        return red(torch.as_tensor(obs[None])).numpy()[0]


redes = {n: cargar(n) for n in ("mejor", "final")}

print("== 1. Estado inicial (auto detenido) ==")
for nombre, red in redes.items():
    elegidas, brechas, ejemplo = [], [], None
    for s in SEMILLAS:
        obs, _ = env.reset(seed=s)
        assert all(np.array_equal(obs[0], obs[i]) for i in range(1, 4)), "los 4 cuadros iniciales no son idénticos"
        v = q(red, obs); o = np.sort(v)
        elegidas.append(NOMBRES_ACCIONES[int(v.argmax())]); brechas.append(o[-1] - o[-2])
        if ejemplo is None:
            ejemplo = v
    print(f"{nombre:6s} pista {SEMILLAS[0]}: " + ", ".join(f"{a} {x:.1f}" for a, x in zip(NOMBRES_ACCIONES, ejemplo)))
    print(f"{'':6s} acción elegida en {len(elegidas)} pistas: {dict(Counter(elegidas))} | "
          f"brecha media entre la 1.ª y la 2.ª acción: {np.mean(brechas):.2f}")

print("\n== 2. Modelo final, greedy desde el estado inicial ==")
obs, _ = env.reset(seed=SEMILLAS[0]); inicial = obs[-1].astype(float)
for k in range(6):
    a = int(q(redes["final"], obs).argmax())
    obs, r, *_ = env.step(a)
    dif = np.abs(obs[-1].astype(float) - inicial).mean()
    print(f"decisión {k}: {NOMBRES_ACCIONES[a]:22s} recompensa {r:+.1f} | diferencia media de píxeles con el cuadro inicial: {dif:.2f}")

print("\n== 3. Arranque forzado: 5 decisiones de «acelerar» y luego greedy (20 pistas) ==")
for nombre, red in redes.items():
    retornos = []
    for s in list(SEMILLAS)[:20]:
        obs, _ = env.reset(seed=s); total, k, fin = 0.0, 0, False
        while not fin:
            a = 3 if k < 5 else int(q(red, obs).argmax())
            obs, r, term, trunc, _ = env.step(a); total += r; k += 1; fin = term or trunc
        retornos.append(total)
    print(f"{nombre:6s} retorno {np.mean(retornos):.1f} ± {np.std(retornos):.1f} (mín {min(retornos):.1f}, máx {max(retornos):.1f})")
