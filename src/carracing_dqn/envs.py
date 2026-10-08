"""
Ambiente CarRacing-v3 y su preprocesamiento.

Cadena de wrappers (en orden):
    CarRacing-v3 (continuous=False)          observación 96x96x3 uint8, 5 acciones discretas
    -> SaltarZoomInicial                     descarta los cuadros de zoom de cámara del inicio
    -> RepetirAccion (frame skip = 4)        repite la acción y suma las recompensas
    -> CortarSinProgreso (solo entrenamiento) corta el episodio si el auto deja de avanzar
    -> PreprocesarCuadro                     recorta el tablero, escala de grises, 84x84
    -> FrameStackObservation (4)             apila los 4 últimos cuadros -> (4, 84, 84) uint8
"""
from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FrameStackObservation

NOMBRES_ACCIONES = ["nada", "girar a la derecha", "girar a la izquierda", "acelerar", "frenar"]


class SaltarZoomInicial(gym.Wrapper):
    """Al reiniciar, CarRacing hace un zoom de cámara de ~50 cuadros en el que el agente no controla nada útil.
    Se avanzan esos cuadros con la acción «nada» para que el primer estado que ve la red ya sea el definitivo."""

    def __init__(self, env, pasos: int = 50):
        super().__init__(env); self.pasos = pasos

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        for _ in range(self.pasos):
            obs, _, term, trunc, info = self.env.step(0)
            if term or trunc:
                obs, info = self.env.reset()
        return obs, info


class RepetirAccion(gym.Wrapper):
    """Frame skip: la misma acción se aplica `k` cuadros seguidos y se suman sus recompensas.
    Cuadros consecutivos son casi idénticos; decidir cada 4 cuadros reduce el costo 4 veces y
    alarga el horizonte efectivo de cada decisión."""

    def __init__(self, env, k: int = 4):
        super().__init__(env); self.k = k

    def step(self, accion):
        total, info = 0.0, {}
        for _ in range(self.k):
            obs, r, term, trunc, info = self.env.step(accion)
            total += r
            if term or trunc:
                break
        return obs, total, term, trunc, info


class CortarSinProgreso(gym.Wrapper):
    """Solo durante el entrenamiento: si pasan `paciencia` decisiones sin recompensa positiva (el auto está
    detenido, girando sobre el pasto o fuera de la pista), el episodio se TRUNCA. No es una terminación real:
    se marca como `truncated` para que el objetivo de Bellman siga estimando el valor futuro."""

    def __init__(self, env, paciencia: int = 100):
        super().__init__(env); self.paciencia = paciencia; self.sin_progreso = 0

    def reset(self, **kw):
        self.sin_progreso = 0
        return self.env.reset(**kw)

    def step(self, accion):
        obs, r, term, trunc, info = self.env.step(accion)
        self.sin_progreso = 0 if r > 0 else self.sin_progreso + 1
        if self.sin_progreso >= self.paciencia and not term:
            trunc = True; info["cortado_sin_progreso"] = True
        return obs, r, term, trunc, info


class PreprocesarCuadro(gym.ObservationWrapper):
    """96x96x3 RGB -> 84x84 escala de grises uint8.
    Se recortan las 12 filas inferiores (tablero con velocidad, giroscopio y puntaje): no son píxeles de la
    pista y el puntaje en pantalla daría a la red una pista espuria del retorno. El color no aporta información
    que el gris no tenga (pista gris, pasto verde, auto rojo se distinguen por intensidad)."""

    def __init__(self, env, tamano: int = 84):
        super().__init__(env); self.tamano = tamano
        self.observation_space = gym.spaces.Box(0, 255, (tamano, tamano), dtype=np.uint8)

    def observation(self, obs):
        gris = cv2.cvtColor(obs[:84, :, :], cv2.COLOR_RGB2GRAY)          # filas 0-83: solo la pista
        return cv2.resize(gris, (self.tamano, self.tamano), interpolation=cv2.INTER_AREA)


def crear_ambiente(cfg_amb: dict, entrenamiento: bool = True, render_mode: str | None = None):
    env = gym.make(cfg_amb["id"], continuous=cfg_amb["continuo"], render_mode=render_mode)
    env = SaltarZoomInicial(env, cfg_amb["pasos_zoom_inicial"])
    env = RepetirAccion(env, cfg_amb["frame_skip"])
    if entrenamiento and cfg_amb.get("paciencia_sin_progreso"):
        env = CortarSinProgreso(env, cfg_amb["paciencia_sin_progreso"])
    env = PreprocesarCuadro(env, cfg_amb["tamano"])
    env = FrameStackObservation(env, cfg_amb["frame_stack"])
    return env
