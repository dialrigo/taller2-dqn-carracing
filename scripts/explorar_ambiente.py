"""
Auditoría del ambiente CarRacing-v3 y línea base aleatoria.

    uv run python scripts/explorar_ambiente.py

1. Imprime los espacios de observación y acción del ambiente original y del ambiente preprocesado.
2. Guarda docs/figuras/preprocesamiento.png (cuadro original vs. los 4 cuadros apilados que ve la red).
3. Evalúa un agente aleatorio en los mismos episodios (semillas) que se usan para evaluar al agente DQN.
"""
import json
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from carracing_dqn.envs import NOMBRES_ACCIONES, crear_ambiente          # noqa: E402
from carracing_dqn.utils import avance_pista, cargar_config, tipo_fin    # noqa: E402

cfg = cargar_config(RAIZ / "configs" / "dqn_carracing.yaml")
ca = cfg["ambiente"]

crudo = gym.make(ca["id"], continuous=False)
print("== Ambiente original ==")
print("Observación:", crudo.observation_space, "| dtype", crudo.observation_space.dtype)
print("Acciones:   ", crudo.action_space, "->", dict(enumerate(NOMBRES_ACCIONES)))
print("Límite de cuadros por episodio:", crudo.spec.max_episode_steps, "| umbral de «resuelto»:", crudo.spec.reward_threshold)
obs, _ = crudo.reset(seed=0)
for _ in range(ca["pasos_zoom_inicial"]):
    obs, *_ = crudo.step(0)
print("Recompensa de un paso sin avanzar:", round(crudo.step(0)[1], 3))
print("Baldosas de la pista (semilla 0):", len(crudo.unwrapped.track), "-> cada baldosa nueva vale", round(1000 / len(crudo.unwrapped.track), 2))

env = crear_ambiente(ca, entrenamiento=False)
print("\n== Ambiente preprocesado (lo que ve la red) ==")
print("Observación:", env.observation_space, "| dtype", env.observation_space.dtype)
o, _ = env.reset(seed=0)
for _ in range(6):
    o, *_ = env.step(3)
print("Forma de un estado:", np.asarray(o).shape, np.asarray(o).dtype)

fig, ax = plt.subplots(1, 5, figsize=(15, 3.3))
crudo.reset(seed=0)
for _ in range(ca["pasos_zoom_inicial"] + 24):
    rgb, *_ = crudo.step(3)
ax[0].imshow(rgb); ax[0].set_title("Original 96x96x3 (RGB)", fontsize=9)
ax[0].axhline(84, color="yellow", lw=1, ls="--"); ax[0].text(2, 94, "tablero recortado", color="yellow", fontsize=7)
for i in range(4):
    ax[i + 1].imshow(np.asarray(o)[i], cmap="gray", vmin=0, vmax=255); ax[i + 1].set_title(f"Cuadro t-{3 - i} (84x84, gris)", fontsize=9)
for a in ax:
    a.axis("off")
plt.suptitle("Preprocesamiento: recorte del tablero, escala de grises, 84x84 y apilamiento de 4 cuadros (cada uno separado por 4 cuadros del simulador)", fontsize=9)
plt.tight_layout(); (RAIZ / "docs" / "figuras").mkdir(parents=True, exist_ok=True)
plt.savefig(RAIZ / "docs" / "figuras" / "preprocesamiento.png", dpi=140, bbox_inches="tight")

print("\n== Agente aleatorio ==")
ce = cfg["evaluacion"]; filas = []
for k in range(ce["episodios"]):
    s = ce["semilla_base"] + k
    obs, _ = env.reset(seed=s); env.action_space.seed(s); ret, fin = 0.0, False
    while not fin:
        obs, r, term, trunc, info = env.step(env.action_space.sample()); ret += r; fin = term or trunc
    filas.append({"retorno": ret, "avance": avance_pista(env), "fin": tipo_fin(env, term, trunc, info)})
R = np.array([f["retorno"] for f in filas]); A = np.array([f["avance"] for f in filas])
res = {"media": float(R.mean()), "desv": float(R.std()), "min": float(R.min()), "max": float(R.max()),
       "avance_pista_medio": float(A.mean()), "episodios": len(R)}
print(json.dumps(res, indent=2, ensure_ascii=False))
(RAIZ / "results").mkdir(exist_ok=True)
json.dump(res, open(RAIZ / "results" / "linea_base_aleatoria.json", "w"), indent=2, ensure_ascii=False)
