# Checklist de revisión por rol

Cada rol responde por un criterio de la rúbrica del Taller 2. Marca cada punto en tu *pull request* (copia la sección de tu rol en la descripción).

| Rol | Criterio de la rúbrica | Peso |
|---|---|---|
| 1 · Integración | Repositorio, organización y coherencia general | transversal |
| 2 · Red neuronal | Red neuronal (dimensiones por capa, justificación, hiperparámetros) | 1,0 |
| 3 · Ambiente | Acciones y observaciones (tipo, dimensiones, dtype, preprocesamiento, recompensas) | 1,0 |
| 4 · Flujo de entrenamiento | Flujo de entrenamiento y particularidades del ambiente (con diagrama) | 1,0 |
| 5 · Experimentos | Reproducibilidad: MLflow, semillas, ejecución | transversal |
| 6 · Resultados | Resultados (métricas concretas, curva, evaluación) | 1,0 |
| 7 · Reflexiones y QA | Reflexión de resultados (0,5) y de lo que más costó (0,5) | 1,0 |

---

## Rol 1 · Integración — Diego Ríos
- [ ] El repositorio es público y el README se ve completo en GitHub (imágenes, GIF, tablas, diagrama).
- [ ] `uv sync` y `uv run python -m carracing_dqn.evaluate --solo-gif` funcionan desde un clon limpio.
- [ ] Cada *pull request* tiene revisión y se integró sin conflictos.
- [ ] No hay archivos personales, rutas locales ni secretos (`git grep -n "C:\\\\Users"` sin resultados).
- [ ] La sección de integrantes del README tiene los 7 nombres y su rol.

## Rol 2 · Red neuronal e hiperparámetros — Nataly Valbuena
- [ ] La tabla de capas del README coincide con `src/carracing_dqn/model.py` (ejecutar `uv run python scripts/resumen_red.py`).
- [ ] Verificar a mano una dimensión: (84 − 8)/4 + 1 = 20 → 20×20 tras la primera convolución; repetir para las otras dos.
- [ ] El número total de parámetros del README coincide con la salida del script.
- [ ] Cada hiperparámetro de `configs/dqn_carracing.yaml` tiene una justificación en el README (sección 5.2). Si alguna te parece débil, mejórala con una fuente (Mnih et al., 2015; Lapan, 2020; documentación de Gymnasium).
- [ ] Explica en tus palabras, en el PR, por qué la normalización `/255` está dentro de la red y por qué la salida es lineal.

## Rol 3 · Ambiente (acciones, observaciones, recompensas) — Integrante 3
- [ ] Ejecutar `uv run python scripts/explorar_ambiente.py` y pegar la salida en el PR.
- [ ] Contrastar la tabla de acciones y el sistema de recompensas del README con la documentación oficial: https://gymnasium.farama.org/environments/box2d/car_racing/
- [ ] Confirmar tipo, forma y dtype: original `Box(0, 255, (96, 96, 3), uint8)`; procesado `(4, 84, 84) uint8`.
- [ ] Revisar `docs/figuras/preprocesamiento.png` y validar que el recorte quita solo el tablero inferior.
- [ ] Revisar que la justificación de apilar 4 cuadros (propiedad de Markov) esté clara en la sección 2.

## Rol 4 · Flujo de entrenamiento y particularidades — Integrante 4
- [ ] **Dibujar a mano** (papel o tableta) el ciclo DQN: ambiente → ε-greedy → memoria de experiencia → minilote → objetivo de Bellman con red objetivo → actualización → sincronización. Guardarlo como `docs/figuras/diagrama_dqn_propio.png` y enlazarlo en la sección 3 del README. En el Taller 1 el profesor pidió esquemas propios, no generados por IA.
- [ ] Recorrer `src/carracing_dqn/train.py` y verificar que cada paso del pseudocódigo del README existe en el código (anotar número de línea en el PR).
- [ ] Revisar la tabla de particularidades (sección 4): ¿falta alguna? (por ejemplo, la penalización de −0,1 por cuadro, o que las baldosas solo pagan la primera vez).

## Rol 5 · Experimentos y reproducibilidad — Integrante 5
- [ ] Clonar el repositorio desde cero, `uv sync`, y correr un entrenamiento corto: `uv run python -m carracing_dqn.train --pasos 3000 --nombre prueba_<tu_nombre>` (≈ 3 min). Pegar la salida en el PR.
- [ ] Abrir MLflow (`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`) y adjuntar una captura de la corrida.
- [ ] Confirmar que la semilla está en la configuración y que el README explica cómo reproducir.
- [ ] No subir tu carpeta `results/prueba_*` ni `mlflow.db` (están en `.gitignore`; verifícalo con `git status`).

## Rol 6 · Resultados — Integrante 6
- [ ] Verificar que **cada cifra** de la sección 6 del README aparece en `results/evaluacion_final.json`, `results/dqn_carracing/evaluaciones.csv` o `results/linea_base_aleatoria.json`. Anotar en el PR la fuente de cada una.
- [ ] Revisar que las figuras de `docs/figuras/` tengan título, ejes con unidades y leyenda legibles.
- [ ] Revisar que las conclusiones de la sección 6.4 no afirmen más de lo que muestran los datos (por ejemplo, una sola semilla).

## Rol 7 · Reflexiones y control de calidad — Integrante 7
- [ ] Recoger la «Bitácora de dificultades» de los *pull requests* de los demás y reescribir la sección 8 del README con esas dificultades reales del grupo.
- [ ] Revisar la sección 7 (reflexión de resultados): debe interpretar el comportamiento del agente y conectar las limitaciones con el diseño del problema.
- [ ] Pasar este checklist completo al final y confirmar que los 7 roles tienen su PR integrado.
- [ ] Leer el README completo de principio a fin buscando errores de redacción o contradicciones.
