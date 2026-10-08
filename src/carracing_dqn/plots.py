"""
Figuras de resultados.   uv run python -m carracing_dqn.plots
Lee results/<nombre>/episodios.csv, evaluaciones.csv, results/evaluacion_episodios.csv y la línea base aleatoria.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .envs import NOMBRES_ACCIONES
from .utils import RAIZ

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3})
AZUL, NAR, VERDE, ROJO, GRIS = "#1f4e79", "#e08e0b", "#2e9e6b", "#c0392b", "#7f8c8d"

ap = argparse.ArgumentParser(); ap.add_argument("--nombre", default="dqn_carracing"); a = ap.parse_args()
R = RAIZ / "results"; F = RAIZ / "docs" / "figuras"; F.mkdir(parents=True, exist_ok=True)
ep = pd.read_csv(R / a.nombre / "episodios.csv"); ev = pd.read_csv(R / a.nombre / "evaluaciones.csv")
base = json.load(open(R / "linea_base_aleatoria.json"))

# 1) Curva de aprendizaje
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
ax[0].scatter(ep.paso, ep.retorno, s=4, color=AZUL, alpha=.25, label="episodio de entrenamiento")
ax[0].plot(ep.paso, ep.retorno.rolling(30, min_periods=5).mean(), color=AZUL, lw=2.2, label="media móvil de 30 episodios")
ax[0].plot(ev.paso, ev.media, "o-", color=NAR, lw=2, ms=5, label="evaluación greedy (5 episodios)")
ax[0].axhline(900, color=VERDE, ls="--", lw=1.3, label="umbral de «resuelto» (900)")
ax[0].axhline(base["media"], color=ROJO, ls=":", lw=1.3, label="agente aleatorio")
ax[0].set_xlabel("Decisiones del agente (× 4 cuadros)"); ax[0].set_ylabel("Retorno"); ax[0].set_title("Retorno por episodio"); ax[0].legend(fontsize=8)
ax[1].plot(ep.paso, 100 * ep.avance_pista.rolling(30, min_periods=5).mean(), color=AZUL, lw=2, label="entrenamiento (media móvil 30)")
ax[1].plot(ev.paso, 100 * ev.avance_pista, "o-", color=NAR, lw=2, ms=5, label="evaluación greedy")
ax[1].set_ylim(0, 105); ax[1].set_xlabel("Decisiones del agente"); ax[1].set_ylabel("% de la pista recorrida"); ax[1].set_title("Avance sobre la pista"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(F / "curva_aprendizaje.png", dpi=150); plt.close()

# 2) Cómo terminan los episodios de entrenamiento
ep["tramo"] = (ep.paso // 25_000 + 1) * 25_000
tipos = ["vuelta completa", "límite de tiempo", "sin progreso", "fuera del mapa"]
comp = ep.groupby("tramo").fin.value_counts(normalize=True).unstack().reindex(columns=tipos).fillna(0) * 100
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
ax[0].stackplot(comp.index, comp.T.values, labels=tipos, colors=[VERDE, NAR, GRIS, ROJO], alpha=.85)
ax[0].set_xlim(comp.index.min(), comp.index.max()); ax[0].set_ylabel("% de episodios del tramo"); ax[0].set_xlabel("Decisiones")
ax[0].set_title("Cómo terminan los episodios de entrenamiento"); ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.17), ncol=4)
d = ep.dropna()
ax[1].plot(d.paso, d.perdida.rolling(30, min_periods=5).mean(), color=AZUL, lw=1.5, label="pérdida de Huber")
ax2 = ax[1].twinx(); ax2.plot(d.paso, d.q_medio.rolling(30, min_periods=5).mean(), color=NAR, lw=1.5, label="Q(s, a) medio"); ax2.grid(False)
ax[1].set_xlabel("Decisiones"); ax[1].set_ylabel("Pérdida", color=AZUL); ax2.set_ylabel("Q medio", color=NAR); ax[1].set_title("Pérdida y valor Q medio de los minilotes")
plt.tight_layout(); plt.savefig(F / "fin_episodios_y_perdida.png", dpi=150, bbox_inches="tight"); plt.close()

# 3) Evaluación final (50 pistas nuevas)
efp = R / "evaluacion_episodios.csv"
if efp.exists():
    ef = pd.read_csv(efp); res = json.load(open(R / "evaluacion_final.json"))
    combos = [("mejor", 0.0, AZUL, "mejor modelo, greedy"), ("mejor", 0.05, "#5dade2", "mejor modelo, ε = 0,05"),
              ("final", 0.05, NAR, "modelo final, ε = 0,05"), ("final", 0.0, ROJO, "modelo final, greedy")]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5), gridspec_kw={"width_ratios": [1.3, 1.1, 1]})
    datos = [ef[(ef.modelo == m) & (ef.epsilon == e)].retorno.values for m, e, _, _ in combos]
    bp = ax[0].boxplot(datos, patch_artist=True, widths=.6)
    for p, (_, _, c, _) in zip(bp["boxes"], combos):
        p.set_facecolor(c); p.set_alpha(.75)
    ax[0].set_xticks(range(1, 5)); ax[0].set_xticklabels([t.replace(", ", "\n") for *_, t in combos], fontsize=8.5)
    ax[0].axhline(900, color=VERDE, ls="--", lw=1.2); ax[0].axhline(base["media"], color=ROJO, ls=":", lw=1.2)
    ax[0].set_ylabel("Retorno"); ax[0].set_title("Evaluación final en 50 pistas nuevas")
    m = ef[(ef.modelo == "mejor") & (ef.epsilon == 0.0)]
    ax[1].scatter(100 * m.avance_pista, m.retorno, color=AZUL, s=24)
    ax[1].set_xlabel("% de la pista recorrida en 1000 cuadros"); ax[1].set_ylabel("Retorno"); ax[1].set_title("Retorno vs. avance (mejor modelo, greedy)")
    x = np.arange(5); w = .38
    for k, (clave, c, et) in enumerate([("mejor_eps0", AZUL, "mejor modelo"), ("final_eps0", ROJO, "modelo final")]):
        d = res[clave]["distribucion_acciones"]
        ax[2].barh(x + (k - .5) * w, [100 * d[n] for n in NOMBRES_ACCIONES], w, color=c, label=et)
    ax[2].set_yticks(x); ax[2].set_yticklabels(NOMBRES_ACCIONES); ax[2].invert_yaxis()
    ax[2].set_xlabel("% de las decisiones (política greedy)"); ax[2].set_title("Acciones elegidas"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(F / "evaluacion_final.png", dpi=150); plt.close()
print("Figuras en", F)
