"""
Evaluación final del agente y visualización de su política.

    uv run python -m carracing_dqn.evaluate                      # modelo mejor y final, 50 episodios
    uv run python -m carracing_dqn.evaluate --solo-gif           # solo regenera el GIF

Salidas: results/evaluacion_final.json, results/evaluacion_episodios.csv,
docs/figuras/politica_episodio.png, docs/agente_carracing.gif
"""
import argparse
import json

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .agent import AgenteDQN
from .envs import NOMBRES_ACCIONES, crear_ambiente
from .utils import RAIZ, avance_pista, cargar_config, tipo_fin

COLORES = ["#95a5a6", "#2e86c1", "#e67e22", "#27ae60", "#c0392b"]


def cargar_agente(ruta, cfg):
    ag = AgenteDQN(cfg["entrenamiento"], n_acciones=5, canales=cfg["ambiente"]["frame_stack"])
    ck = ag.cargar(ruta); return ag, ck


def evaluar_detallado(ag, cfg, semillas, eps=0.0):
    """eps = 0: política greedy pura. eps = 0,05: protocolo de evaluación de Mnih et al. (2015)."""
    env = crear_ambiente(cfg["ambiente"], entrenamiento=False); filas, acciones = [], []
    ag.rng = np.random.default_rng(123)
    for s in semillas:
        obs, _ = env.reset(seed=s); ret, n, fin = 0.0, 0, False
        while not fin:
            a = ag.actuar(obs, eps); acciones.append(a)
            obs, r, term, trunc, info = env.step(a); ret += r; n += 1; fin = term or trunc
        filas.append({"semilla": s, "retorno": ret, "decisiones": n, "avance_pista": avance_pista(env), "fin": tipo_fin(env, term, trunc, info)})
    env.close()
    return pd.DataFrame(filas), np.bincount(acciones, minlength=5) / len(acciones)


def resumen(df):
    r = df.retorno
    return {"media": float(r.mean()), "desv": float(r.std()), "mediana": float(r.median()), "min": float(r.min()), "max": float(r.max()),
            "pct_ge_900": float((r >= 900).mean() * 100), "pct_ge_700": float((r >= 700).mean() * 100),
            "avance_pista_medio": float(df.avance_pista.mean()), "vueltas_completas_pct": float((df.fin == "vuelta completa").mean() * 100),
            "fuera_del_mapa_pct": float((df.fin == "fuera del mapa").mean() * 100), "decisiones_medias": float(df.decisiones.mean())}


def episodio_visual(ag, cfg, semilla, ruta_gif):
    """Recorre un episodio guardando cuadros RGB, valores Q y acciones (para el GIF y la figura de la política)."""
    env = crear_ambiente(cfg["ambiente"], entrenamiento=False, render_mode="rgb_array")
    obs, _ = env.reset(seed=semilla); fin, cuadros, Q, A, R = False, [], [], [], []
    while not fin:
        q = ag.valores_q(obs); a = int(q.argmax())
        cuadros.append(env.render()); Q.append(q); A.append(a)
        obs, r, term, trunc, _ = env.step(a); R.append(r); fin = term or trunc
    env.close()
    imageio.mimsave(ruta_gif, [c[::2, ::2] for c in cuadros[::2]], duration=0.05, loop=0)
    return np.array(cuadros), np.array(Q), np.array(A), np.array(R)


def figura_politica(cuadros, Q, A, R, ruta):
    fig = plt.figure(figsize=(15, 7.5)); g = fig.add_gridspec(3, 6, height_ratios=[1.15, 1, .55])
    idx = np.linspace(5, len(cuadros) - 5, 6).astype(int)
    for j, i in enumerate(idx):
        ax = fig.add_subplot(g[0, j]); ax.imshow(cuadros[i]); ax.axis("off")
        ax.set_title(f"decisión {i}: {NOMBRES_ACCIONES[A[i]]}", fontsize=8.5, color=COLORES[A[i]])
    ax = fig.add_subplot(g[1, :])
    for k in range(5):
        ax.plot(Q[:, k], color=COLORES[k], lw=1.3, label=f"Q(s, {NOMBRES_ACCIONES[k]})")
    for i in idx:
        ax.axvline(i, color="k", ls=":", lw=.8)
    ax.set_ylabel("Valor Q"); ax.legend(fontsize=8, ncol=5, loc="lower center"); ax.set_xlim(0, len(Q))
    ax.set_title(f"Valores Q de las 5 acciones a lo largo de un episodio (retorno {R.sum():.0f})")
    ax = fig.add_subplot(g[2, :], sharex=ax)
    ax.scatter(range(len(A)), A, c=[COLORES[a] for a in A], s=6); ax.set_yticks(range(5)); ax.set_yticklabels(NOMBRES_ACCIONES, fontsize=7)
    ax.set_xlabel("Decisión del agente (cada una = 4 cuadros)"); ax.set_title("Acción elegida (argmax Q)", fontsize=9)
    plt.tight_layout(); plt.savefig(ruta, dpi=140); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(RAIZ / "configs" / "dqn_carracing.yaml"))
    ap.add_argument("--nombre", default="dqn_carracing")
    ap.add_argument("--solo-gif", action="store_true")
    a = ap.parse_args()
    cfg = cargar_config(a.config); ce = cfg["evaluacion"]
    semillas = range(ce["semilla_base"], ce["semilla_base"] + ce["episodios"])
    figs = RAIZ / "docs" / "figuras"; figs.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    mejor, ck = cargar_agente(RAIZ / "models" / f"{a.nombre}_mejor.pt", cfg)
    if not a.solo_gif:
        res, todos = {}, []
        for etiqueta in ["mejor", "final"]:
            ag, ckx = cargar_agente(RAIZ / "models" / f"{a.nombre}_{etiqueta}.pt", cfg)
            for eps in [0.0, 0.05]:
                clave = f"{etiqueta}_eps{eps:g}"
                df, d = evaluar_detallado(ag, cfg, semillas, eps); df["modelo"] = etiqueta; df["epsilon"] = eps; todos.append(df)
                res[clave] = {**resumen(df), "paso": ckx.get("paso"), "epsilon": eps, "distribucion_acciones": dict(zip(NOMBRES_ACCIONES, map(float, d)))}
                print(clave, json.dumps({k: v for k, v in res[clave].items() if k != "distribucion_acciones"}, ensure_ascii=False))
        pd.concat(todos).to_csv(RAIZ / "results" / "evaluacion_episodios.csv", index=False)
        json.dump(res, open(RAIZ / "results" / "evaluacion_final.json", "w"), indent=2, ensure_ascii=False)

    cuadros, Q, A, R = episodio_visual(mejor, cfg, ce["semilla_base"], RAIZ / "docs" / "agente_carracing.gif")
    figura_politica(cuadros, Q, A, R, figs / "politica_episodio.png")
    print("GIF y figura de la política guardados")
