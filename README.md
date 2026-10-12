# Taller 2 · Deep Q-Network para CarRacing-v3 desde píxeles

Simulación y Aprendizaje por Refuerzo · Maestría · Universidad de La Sabana · Grupo 10

<p align="center"><img src="docs/agente_carracing.gif" width="380" alt="Agente DQN conduciendo en CarRacing-v3"></p>

## Resumen

Entrenamos desde cero un agente Double DQN, con la red convolucional de Mnih et al. (2015), para conducir en CarRacing-v3 con acciones discretas a partir de la imagen de la cámara. Con 1 M de cuadros en CPU, el agente obtiene un retorno de 665 ± 172 en 50 pistas que nunca vio, frente a −46 de un agente aleatorio. Recorre en promedio el 77 % de la pista, nunca se sale del mapa y supera 700 puntos en el 54 % de las pistas.

No alcanza el umbral de 900 que define el ambiente como resuelto, porque no completa la vuelta en los 1000 cuadros disponibles. El modelo del final del entrenamiento colapsó (queda inmóvil y, aun forzando el arranque, conduce mucho peor), por lo que se entrega el mejor punto de control.

## Integrantes y roles

Cada integrante revisó la parte del trabajo a su cargo contra el código y los datos. La revisión de cada rol está registrada en su *pull request*.

| Rol | Integrante | Revisión |
|---|---|---|
| 1. Integración del repositorio | Diego Ríos | integró los PR [#1](https://github.com/dialrigo/taller2-dqn-carracing/pull/1) a [#6](https://github.com/dialrigo/taller2-dqn-carracing/pull/6) |
| 2. Red neuronal e hiperparámetros (sección 5) | Nataly Valbuena | [PR #2](https://github.com/dialrigo/taller2-dqn-carracing/pull/2) |
| 3. Ambiente, acciones, observaciones y recompensas (sección 2) | Lorena Valero | [PR #1](https://github.com/dialrigo/taller2-dqn-carracing/pull/1) |
| 4. Flujo de entrenamiento y particularidades (secciones 3 y 4) | Nicolas Gamboa | [PR #3](https://github.com/dialrigo/taller2-dqn-carracing/pull/3) |
| 5. Experimentos y reproducibilidad (sección 1) | Jorge Anaya | [PR #4](https://github.com/dialrigo/taller2-dqn-carracing/pull/4) |
| 6. Resultados (sección 6) | Andrés Gonzalo Díaz Lázaro | [PR #5](https://github.com/dialrigo/taller2-dqn-carracing/pull/5) |
| 7. Reflexiones y control de calidad (secciones 7 y 8) | Andrés Gonzalo Díaz Lázaro | [PR #6](https://github.com/dialrigo/taller2-dqn-carracing/pull/6) |

## Contenido

1. [Cómo ejecutar](#1-cómo-ejecutar)
2. [Acciones, observaciones y recompensas](#2-acciones-observaciones-y-recompensas)
3. [Flujo lógico del entrenamiento](#3-flujo-lógico-del-entrenamiento)
4. [Particularidades del ambiente](#4-particularidades-del-ambiente)
5. [Red neuronal e hiperparámetros](#5-red-neuronal-e-hiperparámetros)
6. [Resultados del entrenamiento](#6-resultados-del-entrenamiento)
7. [Reflexión sobre los resultados](#7-reflexión-sobre-los-resultados)
8. [Reflexión sobre lo que más costó](#8-reflexión-sobre-lo-que-más-costó)
9. [Referencias](#9-referencias)

---

## 1. Cómo ejecutar

El proyecto usa [`uv`](https://docs.astral.sh/uv/) para el entorno (dependencias en `pyproject.toml`, versiones fijadas en `uv.lock`) y [MLflow](https://mlflow.org/) para registrar cada experimento.

```bash
git clone https://github.com/dialrigo/taller2-dqn-carracing.git
cd taller2-dqn-carracing
uv sync                                                    # crea el entorno e instala dependencias

uv run python scripts/explorar_ambiente.py                 # espacios, preprocesamiento y agente aleatorio
uv run python scripts/resumen_red.py                       # arquitectura de la red con dimensiones reales
uv run python scripts/diagnostico_colapso.py               # diagnóstico del colapso del modelo final (sección 7)
uv run python -m carracing_dqn.train                       # entrenamiento completo (usa GPU si existe)
uv run python -m carracing_dqn.train --pasos 3000 --nombre prueba   # verificación rápida (~3 min)
uv run python -m carracing_dqn.train --reanudar            # retoma un entrenamiento interrumpido
uv run python -m carracing_dqn.evaluate                    # evaluación en 50 pistas nuevas y GIF
uv run python -m carracing_dqn.plots                       # figuras de docs/figuras
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # tablero de experimentos en http://127.0.0.1:5000
```

En Windows se recomienda Python 3.12 (`uv sync --python 3.12`), porque con Python 3.14 el paquete `box2d-py` no compila.

Para ejecutarlo sin instalar nada, el notebook [`notebooks/colab_carracing_dqn.ipynb`](notebooks/colab_carracing_dqn.ipynb) corre el mismo código en Google Colab con GPU.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dialrigo/taller2-dqn-carracing/blob/main/notebooks/colab_carracing_dqn.ipynb)

La corrida corta con la semilla 42 (`--pasos 3000`) da las mismas cifras en clones limpios de Windows y Linux: retorno medio −37,5, −37,9 y −42,4 a las 1 000, 2 000 y 3 000 decisiones, y evaluación greedy de −73,9 ± 7,0. El entrenamiento completo, en cambio, no se reproduce bit a bit entre máquinas (sección 6.4).

```
├── configs/dqn_carracing.yaml     # única fuente de hiperparámetros
├── src/carracing_dqn/
│   ├── envs.py                    # ambiente y cadena de preprocesamiento (wrappers)
│   ├── model.py                   # red Q convolucional
│   ├── replay.py                  # memoria de experiencia en uint8 (en disco, reanudable)
│   ├── agent.py                   # ε-greedy, red objetivo, actualización de Bellman (Double DQN)
│   ├── train.py                   # ciclo de entrenamiento, MLflow y puntos de control
│   ├── evaluate.py                # evaluación final, visualización de la política y GIF
│   └── plots.py                   # figuras de resultados
├── scripts/                       # exploración del ambiente, resumen de la red y diagnóstico del colapso
├── notebooks/                     # versión para Google Colab
├── results/                       # registros por episodio, evaluaciones y métricas finales
├── models/                        # modelo entregado (mejor punto de control) y modelo final
└── docs/                          # figuras y GIF del README
```

---

## 2. Acciones, observaciones y recompensas

`CarRacing-v3`, de la familia Box2D de Gymnasium, es una carrera vista desde arriba. El agente controla un auto y debe recorrer una pista que se genera al azar en cada episodio. Es el único ambiente Box2D de Gymnasium cuya observación es una imagen (LunarLander y BipedalWalker entregan vectores), lo que obliga a aprender a reconocer la pista con una red convolucional. Se usa la versión de acciones discretas (`continuous=False`), que es la que admite DQN.

### 2.1 Observaciones

| | Ambiente original | Entrada de la red (tras el preprocesamiento) |
|---|---|---|
| Tipo | `Box(0, 255, (96, 96, 3), uint8)` | `Box(0, 255, (4, 84, 84), uint8)` |
| Contenido | Imagen RGB de la cámara cenital y tablero inferior (velocidad, ABS, giroscopio, volante, puntaje) | 4 cuadros consecutivos en escala de grises, sin tablero |
| Rango | enteros de 0 a 255 | enteros de 0 a 255, divididos por 255 dentro de la red |

![Preprocesamiento](docs/figuras/preprocesamiento.png)

El preprocesamiento está en `src/carracing_dqn/envs.py`. Cada paso tiene una razón concreta.

| Paso | Qué hace | Por qué |
|---|---|---|
| Saltar el zoom inicial | Avanza 50 cuadros con la acción «nada» al reiniciar | Cada episodio arranca con un zoom de cámara en el que la imagen cambia sin que el agente controle nada; la red no debe aprender de esos cuadros. |
| *Frame skip* de 4 | Repite cada acción 4 cuadros y suma las recompensas | Dos cuadros seguidos son casi idénticos (50 FPS). Decidir cada 4 cuadros divide el costo por 4 y hace que cada acción tenga un efecto visible. |
| Recorte del tablero | Elimina las 12 filas inferiores (de 96 a 84 filas) | El tablero no es pista y muestra el puntaje; la red podría correlacionar ese número con el retorno en lugar de aprender a conducir. |
| Escala de grises | Pasa de RGB a 1 canal | Pista (gris), pasto (verde) y bordes (rojo y blanco) se distinguen por intensidad; el color triplica la entrada sin aportar información útil. |
| 84 × 84 | Redimensiona con interpolación por área | Es el tamaño de la arquitectura de Mnih et al. (2015) y deja dimensiones enteras en las convoluciones (20, 9 y 7). |
| Apilar 4 cuadros | El estado son los 4 últimos cuadros procesados | Una sola imagen no es un estado markoviano, porque muestra dónde está el auto pero no hacia dónde ni qué tan rápido va. Con 4 cuadros, que abarcan 16 cuadros del simulador, la red infiere velocidad, dirección y giro, como en los juegos de Atari vistos en clase. |

Al reiniciar, la pila se llena con cuatro copias del primer cuadro, de modo que el estado inicial siempre muestra el auto detenido y sin historia. Ese detalle resulta decisivo en el análisis de la sección 7.

### 2.2 Acciones

El espacio de acciones es `Discrete(5)`, con dtype `int64`.

| Acción | Efecto | Valor que aplica el simulador |
|---|---|---|
| 0 | nada (el auto rueda por inercia) | — |
| 1 | girar a la derecha | dirección 0,6 a la derecha |
| 2 | girar a la izquierda | dirección 0,6 a la izquierda |
| 3 | acelerar | acelerador 0,2 (de 1,0 posible) |
| 4 | frenar | freno 0,8 |

No se puede acelerar y girar en la misma acción, así que el agente debe alternar decisiones y tomar curvas rápidas le resulta más difícil que en la versión continua. Además, la versión discreta usa valores fijos y moderados, por lo que el auto gana velocidad de forma gradual y frenar es comparativamente brusco. Estos valores se verificaron en `gymnasium/envs/box2d/car_racing.py` (Gymnasium 1.4.0).

### 2.3 Recompensas

- Cada cuadro cuesta −0,1 (−0,4 por decisión con *frame skip* de 4), lo que crea una presión permanente para avanzar.
- Cada baldosa nueva de la pista que el auto pisa paga +1000/N, donde N es el número de baldosas de esa pista (cerca de 300, por ejemplo 319 con la semilla 0; es decir, unos 3 puntos por baldosa). Una baldosa solo paga la primera vez.
- Salir del área de juego (|x| o |y| > 333 unidades) da −100, que reemplaza la recompensa de ese cuadro, y termina el episodio.
- Recorrer toda la pista en *k* cuadros da 1000 − 0,1·*k*. El ambiente se considera resuelto con 900 puntos de media (`spec.reward_threshold = 900`), es decir, completando la vuelta en menos de 1000 cuadros.

### 2.4 Terminación del episodio

| Condición | Tipo | Recompensa |
|---|---|---|
| Vuelta completa, al visitar todas las baldosas o al volver a la salida tras visitar más del 95 % (`lap_complete_percent=0.95`) | `terminated` | la acumulada |
| El auto sale del área de juego | `terminated` | −100 |
| Se alcanzan 1000 cuadros (unas 238 decisiones tras el zoom) | `truncated`, límite de tiempo | — |
| Solo en entrenamiento, 100 decisiones seguidas sin recompensa positiva | `truncated`, corte propio | — |

---

## 3. Flujo lógico del entrenamiento

![Flujo lógico del entrenamiento](docs/figuras/diagrama_dqn_propio.png)

El ciclo está implementado en `src/carracing_dqn/train.py` y `agent.py`. En pseudocódigo:

```
inicializar Q_θ (CNN) con pesos aleatorios;  Q_θ⁻ ← Q_θ;  memoria D vacía (80 000)
para paso = 1 … 250 000 decisiones:
    ε ← decaimiento lineal 1,0 → 0,05 durante 100 000 decisiones
    a ← aleatoria si paso ≤ 10 000 (llenado de la memoria); si no, ε-greedy sobre Q_θ(s, ·)
    ejecutar a durante 4 cuadros → s', r (suma), terminated, truncated
    guardar (s, a, r, s', terminated) en D                         # truncated NO se guarda como fin
    cada 4 decisiones:
        muestrear 32 transiciones de D
        a* ← argmax_a' Q_θ(s', a')                                  # la red en línea elige
        y  ← r + 0,99 · (1 − terminated) · Q_θ⁻(s', a*)             # la red objetivo evalúa
        minimizar Huber(Q_θ(s, a), y) con Adam;  ‖∇‖ ≤ 10
    cada 2 500 decisiones:  θ⁻ ← θ
    cada 25 000 decisiones (y en la última): evaluar la política greedy en 5 pistas fijas
        (semillas 10 000 a 10 004) y guardar el mejor modelo
    cada 5 000 decisiones:  punto de control (red, optimizador y memoria en disco) para reanudar
    si terminated o truncated: registrar el episodio en MLflow y reiniciar el ambiente (nueva pista)
guardar el modelo final (además del mejor modelo según la evaluación periódica)
```

Cada línea del pseudocódigo se verificó contra el código; la correspondencia con números de línea está en el [PR #3](https://github.com/dialrigo/taller2-dqn-carracing/pull/3). Tres detalles que el pseudocódigo no hace evidentes:

- ε empieza a bajar desde la primera decisión, así que cuando termina el llenado de la memoria ya vale cerca de 0,90.
- La memoria guarda de *s'* solo el cuadro nuevo, porque los otros 3 ya están en *s*. Así cada transición ocupa 5 cuadros en vez de 8, un 37 % menos de espacio.
- Las 5 pistas de la evaluación periódica son distintas de las 50 de la evaluación final (semillas 1000 a 1049). Elegir el mejor modelo no usa información de la prueba.

Cada componente de DQN cumple una función precisa.

- **Memoria de experiencia.** Los cuadros consecutivos de una vuelta están muy correlacionados. Entrenar con ellos en orden violaría el supuesto de datos independientes del descenso de gradiente y haría que la red olvidara las curvas vistas hace un minuto. Muestrear al azar entre 80 000 transiciones rompe esa correlación y reutiliza cada experiencia: con 32 muestras por cada 4 decisiones nuevas, cada transición se usa en promedio unas 8 veces mientras está en la memoria.
- **Exploración ε-greedy.** Al inicio el agente no sabe que acelerar sobre la pista paga, y la exploración aleatoria genera esas primeras experiencias. ε baja de 1 a 0,05 en 100 000 decisiones y se mantiene en 0,05 para seguir corrigiendo errores.
- **Red objetivo.** Si el objetivo *y* se calculara con la misma red que se actualiza, cada paso de gradiente movería también el objetivo y el valor Q oscilaría. La copia congelada se actualiza cada 2 500 decisiones, es decir, cada 625 pasos de gradiente.
- **Actualización de Bellman con Double DQN.** Q(s, a) se ajusta hacia r + γ·Q(s′, a*). En Double DQN la red en línea elige a* y la red objetivo lo evalúa, lo que reduce la sobreestimación del operador máximo (van Hasselt et al., 2016).

---

## 4. Particularidades del ambiente

| Particularidad | Por qué importa | Cómo se abordó |
|---|---|---|
| Observación en píxeles | El estado no son variables físicas sino una imagen; el agente debe aprender a reconocer la pista. | Red convolucional y preprocesamiento (sección 2.1). |
| Una imagen no es markoviana | Sin historia no se sabe la velocidad ni el sentido del giro. | Apilar 4 cuadros. |
| Pista distinta en cada episodio | No se puede memorizar un recorrido; hay que generalizar a curvas nuevas. | Evaluación en 50 pistas que nunca se usaron en el entrenamiento. |
| Zoom de cámara al inicio | Los primeros 50 cuadros son una transición visual sin control. | Se saltan al reiniciar. |
| Recompensa por baldosa nueva, una sola vez | Volver por la pista ya recorrida no paga y quedarse quieto cuesta −0,1 por cuadro. | Corte del episodio de entrenamiento tras 100 decisiones sin recompensa positiva, que evita llenar la memoria con el auto detenido o girando en el pasto. |
| Salir del área de juego cuesta −100 y termina el episodio | Es la única recompensa grande y negativa. Produce objetivos de Bellman mucho mayores en magnitud que el resto y, si se repite, el agente aprende a evitar los bordes del mapa. | Se marca como `terminated`, porque es un fin real sin futuro que estimar, y la pérdida de Huber limita el gradiente de esos errores grandes. |
| Escala de recompensa variable y sin recortar | Cada baldosa paga 1000/N y N cambia con la pista; una decisión sin avance suma −0,4. A diferencia de Atari, las recompensas no se recortan a [−1, 1]. | Se conserva la escala original para que el retorno sea comparable con el umbral de 900. La pérdida de Huber y el recorte de gradiente (‖∇‖ ≤ 10) mantienen estable el entrenamiento. |
| Menos agarre fuera de la pista | Sobre el pasto la fricción baja al 60 %, el auto derrapa y el efecto de una acción depende de dónde estén las ruedas. | No se modela explícitamente. Los 4 cuadros apilados permiten inferir el derrape y el corte sin progreso descarta los episodios atascados en el pasto. |
| La vuelta completa es un fin real | El episodio termina al visitar todas las baldosas o al volver a la salida tras recorrer más del 95 %. | Se marca como `terminated`, y la evaluación la cuenta como vuelta completa con la señal `lap_finished` del ambiente (`utils.py`, función `tipo_fin`). |
| El corte propio no es un fin real | Si se tratara como terminal, el agente aprendería que detenerse acaba el episodio sin costo. | Se marca como `truncated`, de modo que el objetivo de Bellman sigue estimando el futuro. La evaluación usa el ambiente sin corte. |
| Límite de 1000 cuadros | Tras el zoom quedan unas 238 decisiones; para resolver el ambiente hay que ir rápido. | Se reporta el porcentaje de pista recorrida además del retorno. |
| Acciones exclusivas | No se puede acelerar y girar a la vez. | El agente aprende a alternar ambas acciones; la distribución de acciones se analiza en la sección 6.3. |
| Costo de simulación | Box2D y el renderizado avanzan a unos 125 cuadros por segundo en CPU. | *Frame skip* de 4, memoria en disco y puntos de control para reanudar. |

---

## 5. Red neuronal e hiperparámetros

### 5.1 Arquitectura

La red sigue la arquitectura DQN de Mnih et al. (2015), con la salida reducida a 5 acciones (`src/carracing_dqn/model.py`). La tabla coincide con la salida de `uv run python scripts/resumen_red.py`, y las dimensiones y parámetros se comprobaron a mano.

| Capa | Configuración | Salida (C × H × W) | Parámetros |
|---|---|---|---|
| Entrada | 4 cuadros de 84×84, uint8 dividido por 255 | 4 × 84 × 84 | 0 |
| Conv 1 | 32 filtros, *kernel* 8×8, *stride* 4, ReLU | 32 × 20 × 20 | 8 224 |
| Conv 2 | 64 filtros, *kernel* 4×4, *stride* 2, ReLU | 64 × 9 × 9 | 32 832 |
| Conv 3 | 64 filtros, *kernel* 3×3, *stride* 1, ReLU | 64 × 7 × 7 | 36 928 |
| Aplanar | — | 3 136 | 0 |
| Densa | 3 136 → 512, ReLU | 512 | 1 606 144 |
| Salida | 512 → 5, lineal | 5 valores Q(s, a), uno por acción | 2 565 |
| Total | | | 1 686 693 |

La dimensión de salida de cada convolución es ⌊(entrada − kernel)/stride⌋ + 1, es decir (84 − 8)/4 + 1 = 20, (20 − 4)/2 + 1 = 9 y (9 − 3)/1 + 1 = 7.

El diseño responde a la naturaleza de la entrada.

- **Convoluciones.** La entrada es una imagen. Las convoluciones comparten pesos por posición y detectan los bordes de la pista y la posición del auto donde sea que aparezcan.
- **Strides grandes al inicio.** El *kernel* de 8×8 con *stride* 4 reduce rápido la resolución; el detalle fino de 84×84 no es necesario para decidir y así se reduce el costo.
- **Los 4 cuadros como canales.** Entran como 4 canales de la primera convolución, de modo que la red compara cuadros consecutivos en la misma posición y extrae el movimiento.
- **Capa densa de 512.** Combina las 3 136 características espaciales en una representación de la situación, como una curva cercana, el auto desviado o una velocidad alta.
- **Salida lineal de 5 neuronas.** Una sola pasada da Q para todas las acciones. Los valores Q no están acotados, y una activación sigmoide o tanh los saturaría.
- **Normalización dentro de la red.** La división por 255 ocurre en la primera operación, así la memoria guarda uint8 y ocupa 4 veces menos RAM que en float32.

### 5.2 Hiperparámetros

Todos los valores están en `configs/dqn_carracing.yaml`. La tabla los compara con los de Mnih et al. (2015) y explica cada diferencia.

| Hiperparámetro | Valor | Mnih et al. (2015) | Razón |
|---|---|---|---|
| Decisiones de entrenamiento | 250 000 (1 M de cuadros) | 50 M de cuadros | Presupuesto que cabe en una jornada de CPU (unas 3,6 h). Es 50 veces menos que en Atari, una de las razones por las que el agente no llega al umbral de 900 (sección 7). |
| γ | 0,99 | 0,99 | Horizonte efectivo 1/(1 − γ) de unas 100 decisiones, es decir 400 cuadros u 8 s de simulación a 50 FPS. Cubre de sobra la llegada de una curva (1 a 2 s), de modo que el agente puede valorar frenar o girar antes de salirse. |
| Optimizador y tasa de aprendizaje | Adam, 1 × 10⁻⁴ | RMSProp, 2,5 × 10⁻⁴ | Adam es el optimizador habitual en implementaciones modernas de DQN (Lapan, 2020); con objetivos que se mueven, una tasa menor evita que las convoluciones oscilen. |
| Tamaño del minilote | 32 | 32 | Mismo valor del artículo original; en CPU cada actualización tarda unos 35 ms. |
| Memoria de experiencia | 80 000 transiciones | 1 000 000 | Limitada por la RAM. Cada transición ocupa 35 280 bytes en uint8 (estado de 4×84×84 y cuadro nuevo de 84×84), unos 2,8 GB en total. Cubre unos 340 episodios, suficiente para descorrelacionar los minilotes. |
| Inicio del aprendizaje | 10 000 decisiones | 50 000 cuadros | Llena la memoria con experiencia aleatoria antes de la primera actualización, para no entrenar sobre un puñado de transiciones muy parecidas. |
| Frecuencia de actualización | cada 4 decisiones | cada 4 acciones | Mismo valor del artículo. Equilibra recolección y aprendizaje; con minilotes de 32 cada 4 decisiones, cada transición se muestrea en promedio 32/4 = 8 veces mientras está en la memoria. |
| Sincronización de la red objetivo | cada 2 500 decisiones | cada 10 000 actualizaciones | Equivale a 625 actualizaciones con objetivo fijo. Es más frecuente que en el artículo porque el entrenamiento es 50 veces más corto; con 10 000 el objetivo cambiaría muy pocas veces. |
| ε inicial y final | de 1,0 a 0,05 en 100 000 decisiones | de 1,0 a 0,1 en 1 M de cuadros | Exploración larga, el 40 % del entrenamiento, porque la recompensa positiva exige acelerar varias decisiones seguidas sobre la pista. El valor final de 0,05 es el que el artículo usa al evaluar. |
| Pérdida | Huber | error recortado a [−1, 1] | Equivale al recorte del error del artículo, cuadrática cerca de 0 y lineal lejos, y es robusta a los saltos de −100 al salir del mapa. |
| Recorte del gradiente | norma ≤ 10 | — | Evita actualizaciones explosivas cuando los valores Q crecen. |
| Double DQN | activado | — (van Hasselt et al., 2016) | La red en línea elige la acción y la red objetivo la evalúa, lo que reduce la sobreestimación de Q. |
| Corte sin progreso | 100 decisiones | — | Ahorra cómputo en episodios estancados; solo se usa en entrenamiento. |
| Semilla | 42 | — | Reproducibilidad. |

---

## 6. Resultados del entrenamiento

El entrenamiento cubrió 250 000 decisiones (1 000 000 de cuadros y 1 186 episodios) en una CPU de 2 núcleos, durante unas 3,6 horas, con la semilla 42. Todas las cifras de esta sección salen de los archivos de `results/`.

### 6.1 Curva de aprendizaje

![Curva de aprendizaje](docs/figuras/curva_aprendizaje.png)

| Decisiones | Evaluación greedy (5 pistas fijas) | Pista recorrida |
|---|---|---|
| 25 000 | −73,7 | 7 % |
| 50 000 | 45,9 | 15 % |
| 75 000 | −33,3 | 7 % |
| 100 000 | 379,7 | 48 % |
| 125 000 | 492,9 | 59 % |
| 150 000 | 399,1 | 50 % |
| 175 000 | −95,0 (atascado) | 1 % |
| 200 000 | 655,6 (mejor punto de control) | 76 % |
| 225 000 | 518,8 | 62 % |
| 250 000 | −95,0 (atascado) | 1 % |

El aprendizaje pasa por tres fases.

1. **Exploración, de 0 a 80 000 decisiones.** Hasta la decisión 58 000, aproximadamente, la media móvil de 30 episodios se mantiene entre −42 y 0, al nivel del agente aleatorio (−46); después sube despacio hasta cerca de 50. Entre las decisiones 10 000 y 80 000, entre el 61 % y el 81 % de los episodios de cada tramo de 10 000 decisiones terminan por el corte sin progreso, porque el auto apenas avanza.
2. **Despegue, de 80 000 a 130 000.** Con ε cayendo hacia 0,05, el agente aprende a acelerar sobre la pista y a corregir en las curvas. La media móvil sube de cerca de 50 a cerca de 580, y la proporción de episodios cortados sin progreso baja de alrededor del 64 % al 5 %. El mejor episodio de entrenamiento, con 917,6 puntos y la vuelta completa, ocurre en la decisión 131 158.
3. **Meseta inestable, de 130 000 a 250 000.** La media móvil oscila entre 310 y 605. El 93 % de los episodios acaba por límite de tiempo y solo el 0,6 % por salir del mapa; el agente ya no se sale, pero no alcanza a completar la vuelta en 1000 cuadros.

![Fin de episodios y pérdida](docs/figuras/fin_episodios_y_perdida.png)

La pérdida de Huber sube mientras el agente aprende, con una media móvil que pasa de 0,2 al inicio a un máximo de 2,9 cerca de la decisión 200 000. No es una señal de error. Los objetivos de Bellman crecen a medida que el agente encuentra recompensas, y el valor Q medio de los minilotes pasa de cerca de 0 a cerca de 104.

### 6.2 Evaluación final en 50 pistas nuevas

La evaluación usa las semillas 1000 a 1049, que no se usaron en el entrenamiento ni para elegir el modelo. Se evalúan el mejor punto de control (decisión 200 000) y el modelo final (decisión 250 000), con política greedy (ε = 0) y con ε = 0,05, que es el protocolo de evaluación de Mnih et al. (2015).

| Agente | Retorno medio ± desv. | Mediana | Mín. – máx. | Pista recorrida | ≥ 700 pts | Vueltas completas | Fuera del mapa |
|---|---|---|---|---|---|---|---|
| Mejor modelo, greedy (`models/dqn_carracing_mejor.pt`) | 665,0 ± 171,9 | 713,5 | 273,9 – 912,7 | 76,6 % | 54 % | 4 % | 0 % |
| Mejor modelo, ε = 0,05 | 635,0 ± 221,9 | 675,5 | 62,0 – 899,7 | 73,8 % | 46 % | 2 % | 2 % |
| Modelo final, ε = 0,05 | −48,8 ± 36,5 | −57,3 | −95,0 – 40,0 | 5,3 % | 0 % | 0 % | 0 % |
| Modelo final, greedy | −95,0 ± 0,0 | −95,0 | −95,0 – −95,0 | 0,7 % | 0 % | 0 % | 0 % |
| Agente aleatorio (referencia) | −46,0 ± 18,6 | — | −81,3 – −5,3 | 5,6 % | 0 % | 0 % | — |

![Evaluación final](docs/figuras/evaluacion_final.png)

- **El mejor modelo conduce.** Recorre en promedio el 76,6 % de pistas que nunca vio, nunca se sale del mapa y supera 700 puntos en el 54 % de ellas. Completa la vuelta en 2 de 50 pistas, con un máximo de 912,7 puntos.
- **La selección del modelo no infló el resultado.** El mejor punto de control se eligió con 5 pistas fijas (655,6) y en 50 pistas nuevas obtiene 665,0, una cifra equivalente.
- **El retorno es una función casi exactamente lineal del avance.** El ajuste es retorno ≈ 1 002 × (fracción de pista) − 103, con R² = 0,9998 en los 50 episodios. El límite no es que el agente se equivoque de camino, sino que no le alcanza el tiempo: 48 de 50 episodios terminan por el tope de 1000 cuadros.
- **El modelo final colapsó.** Con política greedy queda inmóvil en las 50 pistas (−95 equivale a −0,1 por cuadro durante los 950 cuadros del episodio); con ε = 0,05 rinde igual que un agente aleatorio. La sección 7 explica por qué.
- **No se alcanzó el umbral de 900.** El agente entregado obtiene 665 de media, unos 711 puntos por encima del agente aleatorio.

### 6.3 Visualización de la política

![Política durante un episodio](docs/figuras/politica_episodio.png)

En un episodio completo (pista 1000, retorno de 816) el agente acelera en las rectas, alterna giros cortos en las curvas (a la izquierda en la curva cerrada de la decisión 141) y casi nunca frena. En los 50 episodios de evaluación, el mejor modelo elige acelerar en el 62 % de las decisiones, girar a la izquierda en el 22 %, girar a la derecha en el 14 %, y «nada» o frenar en menos del 2 %. Los valores Q de las 5 acciones están muy cerca entre sí la mayor parte del tiempo y solo se separan en las curvas. Las caídas fuertes, entre las decisiones 200 y 220, coinciden con tramos en los que el auto se sale del borde de la pista y pierde recompensa.

### 6.4 Conclusiones de los resultados

1. Con un DQN implementado desde cero y 1 M de cuadros en CPU, el agente aprende a conducir desde píxeles. Pasa de −46 (aleatorio) a 665 ± 172 en pistas nuevas y recorre tres cuartas partes de la pista sin salirse.
2. El desempeño queda por debajo del umbral de 900 principalmente por velocidad. Ningún episodio greedy termina fuera del mapa y 48 de 50 se agotan por tiempo. Las salidas del borde de la pista que sí ocurren (sección 6.3) cuestan tiempo, pero no terminan el episodio.
3. El entrenamiento es inestable. Dos de las diez evaluaciones periódicas, en las decisiones 175 000 y 250 000, encontraron al agente inmóvil. Guardar el mejor punto de control fue indispensable, porque el modelo final no sirve.
4. El resultado no es reproducible bit a bit entre máquinas. La corrida corta de 3 000 decisiones da exactamente las mismas cifras en Windows y Linux con la semilla 42. En cambio, una segunda corrida completa, hecha en otro equipo con la misma configuración y semilla, obtuvo su mejor punto de control en la decisión 125 000, con 554 ± 210 en las 50 pistas de evaluación frente a 665 ± 172 de la corrida reportada (detalle en el [PR #4](https://github.com/dialrigo/taller2-dqn-carracing/pull/4)). Ambas corridas aprenden a conducir y ninguna resuelve el ambiente, pero la diferencia de unos 110 puntos muestra que 665 es una realización y no un valor esperado. Con dos corridas no se puede estimar la variabilidad con rigor.

---

## 7. Reflexión sobre los resultados

### Qué aprendió el agente

A partir de solo 4 imágenes en gris, la red aprendió a distinguir la pista del pasto, a anticipar curvas y a alternar la aceleración con giros cortos. Prefiere acelerar casi siempre y no frena. Frenar no da recompensa inmediata y su beneficio, entrar a la curva sin salirse, llega varias decisiones después; por eso es la acción que más le cuesta valorar a DQN con γ = 0,99 y un presupuesto corto de entrenamiento.

### Por qué no llega a 900

El ambiente se resuelve recorriendo toda la pista en menos de 1000 cuadros. Nuestro agente mantiene una velocidad prudente y alterna giros, de modo que completa la vuelta solo en el 4 % de las pistas. Tres decisiones de diseño lo limitan.

- **Acciones exclusivas.** No puede acelerar y girar a la vez, como sí puede hacerlo en la versión continua. Cada corrección en una curva es una decisión sin acelerar.
- **El zoom inicial.** Saltarlo consume 50 de los 1000 cuadros disponibles.
- **El presupuesto.** 1 M de cuadros es 50 veces menos que los 50 M por juego de Mnih et al. (2015). La curva de la sección 6.1 no muestra que el agente haya agotado lo que puede aprender; muestra que en ese presupuesto no se estabilizó.

### El colapso del modelo final

El colapso combina dos fallas distintas. Lo verificamos con `scripts/diagnostico_colapso.py`, que carga los dos modelos entregados.

La primera es una trampa en el estado inicial. Tras el reinicio, los 4 cuadros apilados son idénticos, con el auto detenido, y los valores Q de las 5 acciones quedan casi empatados. El mejor modelo elige acelerar en las 50 pistas de evaluación, con una ventaja media de solo 1,7 puntos sobre la segunda acción (en la pista 1000, 61,7 frente a 59,5 de «nada»). El modelo final elige girar en las 50 pistas, con una ventaja media de 0,18 puntos (en la pista 1000, 93,4 para girar a la izquierda frente a 91,7 para acelerar). Girar con el auto detenido no lo mueve. La imagen siguiente difiere del cuadro inicial en 0,01 niveles de gris de media, apenas las ruedas, y la red vuelve a ver el mismo estado y elige lo mismo. La política greedy queda en un bucle determinista y el auto nunca arranca, lo que da −0,1 por cuadro durante 950 cuadros, es decir −95 en todas las pistas.

La segunda es que el modelo final conduce peor. Si se fuerzan 5 decisiones de aceleración al inicio y luego se deja actuar a la red, en las primeras 20 pistas de evaluación el mejor modelo obtiene 615 ± 202 y el modelo final solo 107 ± 25. La trampa explica el −95, pero aun sacándolo de ella el modelo final perdió buena parte de lo que había aprendido. Entre las decisiones 200 000 y 250 000 el entrenamiento no solo cambió el orden de las acciones en un estado poco frecuente, sino que degradó la política en general.

Estas fallas se conectan con el diseño del problema.

- **El estado inicial es poco frecuente en la memoria de experiencia.** Aparece una vez por episodio frente a unos 237 estados en movimiento, así que la red casi no recibe gradiente sobre él y su valor deriva con el resto del entrenamiento. Con empates de décimas, cualquier deriva cambia la acción elegida.
- **El corte sin progreso oculta el problema durante el entrenamiento.** Un episodio atascado se corta a las 100 decisiones y la exploración con ε = 0,05 a veces lo saca, de modo que la curva de entrenamiento no muestra el colapso. Solo la evaluación greedy periódica lo detectó, dos veces, en las decisiones 175 000 y 250 000.
- **Los valores Q siguen creciendo mientras el retorno se estanca.** El Q medio de los minilotes pasa de cerca de 20 a cerca de 104 entre las decisiones 150 000 y 250 000. Eso no prueba por sí solo una sobreestimación. El mejor modelo gana unos 2,8 puntos por decisión (665/237), y con γ = 0,99 y el límite de tiempo tratado como truncamiento, ese ritmo vale 2,8/(1 − 0,99), unos 280; un Q de 104 puede ser un valor que aún se está propagando. Lo que sí importa es la escala relativa. Q crece hasta cerca de 90 en el estado inicial mientras la diferencia entre acciones sigue en décimas, de modo que un error pequeño en el ajuste basta para cambiar la acción. Distinguir sobreestimación de propagación lenta requeriría comparar Q con el retorno descontado observado en cada estado.
- **Dos corridas, dos resultados.** La segunda corrida completa, con la misma semilla en otra máquina, eligió su mejor punto de control en la decisión 125 000 y obtuvo 554 ± 210. Que dos corridas difieran tanto, y que ambas tengan su mejor modelo antes del final, indica que la inestabilidad es del método con este presupuesto y no un accidente de una corrida.

### Qué haríamos distinto

1. Seleccionar el modelo con evaluaciones greedy más frecuentes y con más pistas, y descartar los puntos de control que se atasquen. Esta práctica fue la que permitió entregar un modelo funcional.
2. Arrancar el episodio con el auto en movimiento, con unas decisiones fijas de aceleración tras el zoom. El diagnóstico muestra que esto elimina el −95, pero no recupera el modelo final (107), así que no sustituye la selección del mejor punto de control.
3. Usar una tasa de aprendizaje decreciente al final y entrenar con más cuadros en GPU.
4. Probar *Dueling DQN*, que separa el valor del estado de la ventaja de cada acción y es útil cuando las acciones valen casi lo mismo, como en el estado inicial, y *Prioritized Experience Replay*, para dar más peso a estados poco frecuentes.
5. Repetir con varias semillas para estimar la variabilidad. La diferencia entre nuestras dos corridas completas (665 frente a 554) muestra que hace falta.

---

## 8. Reflexión sobre lo que más costó

Esta sección reúne las dificultades que cada integrante registró en su *pull request* ([#1](https://github.com/dialrigo/taller2-dqn-carracing/pull/1) a [#6](https://github.com/dialrigo/taller2-dqn-carracing/pull/6)) y las que aparecieron al construir y entrenar el agente.

### Dificultades técnicas

1. **El costo de cómputo y las interrupciones.** CarRacing simula a unos 125 cuadros por segundo en CPU y cada actualización de la red convolucional tarda unos 35 ms. Un millón de cuadros tomó cerca de 3,6 horas, sin margen para buscar hiperparámetros ni repetir semillas. El proceso se interrumpió en las primeras horas, así que implementamos puntos de control cada 5 000 decisiones con la memoria de experiencia en disco. El *frame skip* y guardar solo el cuadro nuevo de *s′* (unos 35 KB por transición en lugar de 56 KB) fueron necesarios para que la memoria cupiera en la máquina.
2. **La documentación del ambiente no alcanza.** La página de Gymnasium no detalla la magnitud de las acciones discretas (dirección de ±0,6, acelerador de 0,2 y freno de 0,8) ni la regla que da por completa la vuelta al 95 %, y tuvimos que leer `car_racing.py` para confirmarlas. En el mismo proceso, la figura de preprocesamiento parecía correcta, y solo al comparar píxeles vimos que el cuadro original y los apilados eran de instantes distintos, por la forma en que el script interactuaba con el zoom inicial.
3. **Reproducir no es lo mismo que fijar una semilla.** La corrida corta da cifras idénticas en distintos equipos, pero la reproducción completa en otra máquina, con la misma semilla, terminó en 554 ± 210 en lugar de 665 ± 172. Reproducir también exigió resolver incompatibilidades. `box2d-py` no compila con Python 3.14 en Windows, y con PyTorch 2.6 o superior `torch.load` dejó de abrir los puntos de control, de modo que `--reanudar` fallaba hasta indicar `weights_only=False`.

### Dificultades conceptuales

4. **Por qué una imagen no basta.** Una imagen muestra dónde está el auto, pero no hacia dónde va ni qué tan rápido. Relacionarlo con la propiedad de Markov vista en clase fue lo que justificó la entrada de 4 canales.
5. **Terminación frente a truncamiento.** El corte por falta de progreso y el límite de 1000 cuadros no son fines reales del episodio. Si se marcan como terminales, el objetivo de Bellman anula el valor futuro y el agente aprende que detenerse no cuesta nada. Es una sola línea de código, `terminated` frente a `truncated`, con consecuencias en todo el aprendizaje.
6. **Adaptar los hiperparámetros de Atari.** Con 50 veces menos entrenamiento que Mnih et al. (2015), valores como una memoria de 1 M de transiciones o sincronizar la red objetivo cada 10 000 pasos no aplican, y tuvimos que justificar cada ajuste en lugar de copiarlo.

### Lo que más nos costó, verificar

7. **Las cifras intuitivas resultaron falsas con frecuencia.** El borrador decía que cada transición se reutilizaba unas 2 veces; al calcularlo a mano, en dos revisiones independientes, el resultado fue 8. La revisión de resultados encontró rangos de la curva leídos a ojo que no coincidían con los datos y un porcentaje mal redondeado. Los valores Q del estado inicial citados en el borrador de la sección 7 no coincidían con los de los modelos entregados, y la afirmación de que el modelo final no conducía peor resultó falsa al probarla (107 frente a 615 con arranque forzado).
8. **Interpretar curvas que se contradicen.** La pérdida subía mientras el agente mejoraba, y la curva de entrenamiento se veía estable mientras el modelo greedy quedaba inmóvil. Entenderlo exigió mirar el comportamiento del agente, con los valores Q en el estado inicial, la distribución de acciones y el arranque forzado, y no solo las métricas agregadas. La primera explicación del colapso era plausible pero incompleta; solo un experimento dirigido (`scripts/diagnostico_colapso.py`) mostró que se trataba de dos fallas.

En síntesis, entrenar el agente fue costoso, pero lo que más nos costó fue no dar por cierta una explicación o una cifra hasta haberla comprobado contra el código o los datos.

---

## 9. Referencias

- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529–533.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI*.
- Wang, Z., Schaul, T., Hessel, M., et al. (2016). Dueling network architectures for deep reinforcement learning. *ICML*.
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. *ICLR*.
- Lapan, M. (2020). *Deep Reinforcement Learning Hands-On* (2.ª ed.). Packt.
- Farama Foundation. *Gymnasium, Car Racing*. https://gymnasium.farama.org/environments/box2d/car_racing/
- PyTorch. *Reinforcement Learning (DQN) Tutorial*. https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

## Uso de IA generativa

El código base y el borrador de este documento se construyeron con apoyo de un asistente de IA (Claude, de Anthropic). Las cifras provienen de la ejecución real del código de este repositorio, y cada integrante verificó la sección de su rol contra el código y los datos, como consta en su *pull request*.
