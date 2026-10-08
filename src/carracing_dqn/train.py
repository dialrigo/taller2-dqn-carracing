"""
Entrenamiento del agente DQN en CarRacing-v3.

    uv run python -m carracing_dqn.train --config configs/dqn_carracing.yaml
    uv run python -m carracing_dqn.train --config configs/dqn_carracing.yaml --pasos 20000 --nombre piloto
    uv run python -m carracing_dqn.train --config configs/dqn_carracing.yaml --reanudar     # retoma tras un corte

Salidas: results/<nombre>/episodios.csv, evaluaciones.csv, config.yaml; models/<nombre>_mejor.pt;
registro de parámetros y métricas en MLflow (base mlflow.db, ver con `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`).
"""
import argparse
import copy
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml

from .agent import AgenteDQN
from .envs import crear_ambiente
from .replay import MemoriaExperiencia
from .utils import RAIZ, aplanar, avance_pista, cargar_config, fijar_semillas, tipo_fin


def leer_csv(ruta):
    try:
        return pd.read_csv(ruta).to_dict("records")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return []


def evaluar(agente, cfg_amb, episodios: int, semilla_base: int):
    """Política greedy (ε = 0) SIN el corte por falta de progreso: es el ambiente original con preprocesamiento."""
    env = crear_ambiente(cfg_amb, entrenamiento=False)
    filas = []
    for k in range(episodios):
        obs, _ = env.reset(seed=semilla_base + k)
        ret, pasos, fin = 0.0, 0, False
        while not fin:
            obs, r, term, trunc, info = env.step(agente.actuar(obs, 0.0))
            ret += r; pasos += 1; fin = term or trunc
        filas.append({"semilla": semilla_base + k, "retorno": ret, "decisiones": pasos,
                      "avance_pista": avance_pista(env), "fin": tipo_fin(env, term, trunc, info)})
    env.close()
    return pd.DataFrame(filas)


def entrenar(cfg: dict, nombre: str, reanudar: bool = False):
    ce, ca = cfg["entrenamiento"], cfg["ambiente"]
    out = RAIZ / "results" / nombre; out.mkdir(parents=True, exist_ok=True)
    (RAIZ / "models").mkdir(exist_ok=True); (RAIZ / "checkpoints").mkdir(exist_ok=True)
    ck_ruta = RAIZ / "checkpoints" / f"{nombre}.pt"
    yaml.safe_dump(cfg, open(out / "config.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    fijar_semillas(ce["semilla"])

    env = crear_ambiente(ca, entrenamiento=True)
    agente = AgenteDQN(ce, n_acciones=env.action_space.n, canales=ca["frame_stack"], semilla=ce["semilla"])
    memoria = MemoriaExperiencia(ce["memoria"], (ca["frame_stack"], ca["tamano"], ca["tamano"]), ce["semilla"] + 1,
                                 carpeta=RAIZ / "checkpoints" / f"{nombre}_memoria")
    env.action_space.seed(ce["semilla"])

    paso0, log_ep, log_ev, mejor = 1, [], [], -np.inf
    if reanudar and ck_ruta.exists():
        ck = torch.load(ck_ruta, map_location=agente.dev)
        agente.q.load_state_dict(ck["red"]); agente.q_obj.load_state_dict(ck["red_objetivo"]); agente.opt.load_state_dict(ck["opt"])
        paso0, mejor = ck["paso"] + 1, ck["mejor"]
        if "memoria" in ck:
            memoria.restaurar_estado(ck["memoria"])
        log_ep, log_ev = leer_csv(out / "episodios.csv"), leer_csv(out / "evaluaciones.csv")
        log_ep = [f for f in log_ep if f["paso"] < paso0]; log_ev = [f for f in log_ev if f["paso"] < paso0]
        print(f"Reanudando desde el paso {paso0:,} con {len(memoria):,} transiciones en memoria")

    mlflow.set_tracking_uri(f"sqlite:///{RAIZ / cfg['mlflow']['uri']}")
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    with mlflow.start_run(run_name=nombre):
        mlflow.log_params(aplanar(cfg))
        obs, _ = env.reset(seed=ce["semilla"] + paso0)
        ret, n_ep, perd, qs, t0 = 0.0, 0, [], [], time.time()
        inicio_aprendizaje = ce["inicio_aprendizaje"] if len(memoria) >= ce["inicio_aprendizaje"] or paso0 == 1 else paso0 + ce["inicio_aprendizaje"]
        for paso in range(paso0, ce["pasos_totales"] + 1):
            eps = agente.epsilon(paso)
            a = env.action_space.sample() if paso <= inicio_aprendizaje else agente.actuar(obs, eps)
            obs2, r, term, trunc, info = env.step(a)
            memoria.agregar(obs, a, r, obs2, term)                     # solo `term` anula el bootstrap
            obs = obs2; ret += r; n_ep += 1

            if paso > inicio_aprendizaje and paso % ce["frecuencia_entrenamiento"] == 0:
                m = agente.actualizar(memoria.muestrear(ce["batch"])); perd.append(m["perdida"]); qs.append(m["q_medio"])
            if paso % ce["sincronizar_objetivo"] == 0:
                agente.sincronizar()

            if term or trunc:
                fila = {"episodio": len(log_ep), "paso": paso, "retorno": ret, "decisiones": n_ep, "epsilon": eps,
                        "avance_pista": avance_pista(env), "fin": tipo_fin(env, term, trunc, info),
                        "perdida": float(np.mean(perd)) if perd else np.nan, "q_medio": float(np.mean(qs)) if qs else np.nan}
                log_ep.append(fila)
                mlflow.log_metrics({"retorno_episodio": ret, "avance_pista": fila["avance_pista"], "epsilon": eps}, step=paso)
                obs, _ = env.reset(); ret, n_ep, perd, qs = 0.0, 0, [], []

            if paso % ce["evaluar_cada"] == 0 or paso == ce["pasos_totales"]:
                ev = evaluar(agente, ca, ce["episodios_evaluacion"], semilla_base=10_000)
                fila = {"paso": paso, "media": ev.retorno.mean(), "desv": ev.retorno.std(), "min": ev.retorno.min(),
                        "max": ev.retorno.max(), "avance_pista": ev.avance_pista.mean()}
                log_ev.append(fila)
                mlflow.log_metrics({"eval_retorno_medio": fila["media"], "eval_avance_pista": fila["avance_pista"]}, step=paso)
                if fila["media"] > mejor:
                    mejor = fila["media"]; agente.guardar(RAIZ / "models" / f"{nombre}_mejor.pt", {"paso": paso, "eval_media": float(mejor), "config": cfg})
                ult = pd.DataFrame(log_ep[-20:]).retorno.mean() if log_ep else float("nan")
                print(f"[{nombre}] paso {paso:>7,} | ε={eps:.3f} | episodios {len(log_ep):>4} | retorno medio últimos 20: {ult:7.1f} | "
                      f"eval greedy: {fila['media']:7.1f} ± {fila['desv']:5.1f} (avance {fila['avance_pista']:.0%}) | {(time.time() - t0) / 60:5.1f} min", flush=True)

            if paso % ce.get("guardar_cada", 5000) == 0:            # punto de control para reanudar tras un corte
                pd.DataFrame(log_ep).to_csv(out / "episodios.csv", index=False)
                pd.DataFrame(log_ev).to_csv(out / "evaluaciones.csv", index=False)
                torch.save({"red": agente.q.state_dict(), "red_objetivo": agente.q_obj.state_dict(), "opt": agente.opt.state_dict(),
                            "paso": paso, "mejor": mejor, "memoria": memoria.guardar_estado()}, str(ck_ruta) + ".tmp")
                Path(str(ck_ruta) + ".tmp").replace(ck_ruta)

        agente.guardar(RAIZ / "models" / f"{nombre}_final.pt", {"paso": ce["pasos_totales"], "config": cfg})
        pd.DataFrame(log_ep).to_csv(out / "episodios.csv", index=False)
        pd.DataFrame(log_ev).to_csv(out / "evaluaciones.csv", index=False)
        json.dump({"duracion_min": (time.time() - t0) / 60, "episodios": len(log_ep), "mejor_eval": float(mejor),
                   "dispositivo": str(agente.dev)}, open(out / "resumen_entrenamiento.json", "w"), indent=2)
        mlflow.log_artifacts(str(out), artifact_path="resultados")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entrena DQN en CarRacing-v3")
    ap.add_argument("--config", default=str(RAIZ / "configs" / "dqn_carracing.yaml"))
    ap.add_argument("--pasos", type=int, default=None, help="sobrescribe entrenamiento.pasos_totales")
    ap.add_argument("--nombre", default="dqn_carracing")
    ap.add_argument("--reanudar", action="store_true")
    a = ap.parse_args()
    cfg = cargar_config(a.config)
    if a.pasos:
        cfg = copy.deepcopy(cfg); cfg["entrenamiento"]["pasos_totales"] = a.pasos
        cfg["entrenamiento"]["evaluar_cada"] = min(cfg["entrenamiento"]["evaluar_cada"], max(a.pasos // 4, 1000))
    if torch.cuda.is_available():
        print("Usando GPU:", torch.cuda.get_device_name(0))
    entrenar(cfg, a.nombre, a.reanudar)
