# Validación de la red neuronal y los hiperparámetros (Rol 2)

Registro de la revisión de la sección 5 del README. Responsable: Nataly Valbuena (`natavalbuenal`).

## Qué revisé
- `src/carracing_dqn/model.py` contra la tabla de capas de §5.1.
- `configs/dqn_carracing.yaml` contra la tabla de hiperparámetros de §5.2.
- Comparación de cada hiperparámetro con la tabla de hiperparámetros de Mnih et al. (2015, *Extended Data Table 1*).

## Salida de `uv run python scripts/resumen_red.py`
```
Capa       Salida (C, H, W)       Parámetros
Conv2d     (32, 20, 20)                8,224
Conv2d     (64, 9, 9)                 32,832
Conv2d     (64, 7, 7)                 36,928
Flatten    (3136,)                         0
Linear     (512,)                  1,606,144
Linear     (5,)                        2,565
Total                              1,686,693
```
Coincide con la tabla de §5.1 capa por capa y en el total.

## Cálculo a mano
Dimensión de salida de una convolución sin relleno: ⌊(entrada − kernel) / stride⌋ + 1.

| Capa | Cálculo | Salida |
|---|---|---|
| Conv 1 | (84 − 8) / 4 + 1 = 20 | 32 × 20 × 20 |
| Conv 2 | (20 − 4) / 2 + 1 = 9 | 64 × 9 × 9 |
| Conv 3 | (9 − 3) / 1 + 1 = 7 | 64 × 7 × 7 |
| Aplanar | 64 · 7 · 7 | 3 136 |

Parámetros (pesos + sesgos):

| Capa | Cálculo | Parámetros |
|---|---|---|
| Conv 1 | 32 · (4 · 8 · 8) + 32 | 8 224 |
| Conv 2 | 64 · (32 · 4 · 4) + 64 | 32 832 |
| Conv 3 | 64 · (64 · 3 · 3) + 64 | 36 928 |
| Densa | 512 · 3 136 + 512 | 1 606 144 |
| Salida | 5 · 512 + 5 | 2 565 |
| **Total** | | **1 686 693** |

El 95 % de los parámetros está en la capa densa: las convoluciones son baratas porque comparten pesos.

## Por qué la normalización ÷255 está dentro de la red
La memoria de experiencia guarda los cuadros como `uint8` (1 byte por píxel). Si se normalizaran antes de guardarlos, cada píxel ocuparía 4 bytes en `float32` y las 80 000 transiciones pasarían de ≈ 2,8 GB a ≈ 11 GB. Al dividir por 255 en la primera operación de la red, la conversión se hace solo sobre el minilote de 32 estados, y la red recibe valores entre 0 y 1, que mantienen estables los gradientes de las convoluciones.

## Por qué la salida es lineal
La red estima Q(s, a), el retorno esperado, que no está acotado: en este ambiente va de ≈ −100 a ≈ 900. Una sigmoide (0 a 1) o una tanh (−1 a 1) no podrían representar esos valores y saturarían los gradientes; una softmax convertiría las salidas en probabilidades, que no es lo que DQN aprende. Con 5 neuronas lineales, una sola pasada entrega el valor de las 5 acciones y la política elige el argmax.

## Cambios realizados en el README
1. **§5.2:** se agrega una columna con el valor de Mnih et al. (2015) para cada hiperparámetro y se justifica cada diferencia (memoria limitada por RAM, sincronización más frecuente porque el entrenamiento es 50 veces más corto, Adam en lugar de RMSProp, Huber como equivalente del recorte del error).
2. **γ:** se precisa el horizonte efectivo: 1/(1 − 0,99) = 100 decisiones = 400 cuadros ≈ 8 s de simulación (el texto anterior hablaba de 1–2 s, que es el tiempo de llegada de una curva, no el horizonte).
3. **Reutilización de la memoria (§3 y §5.2):** el README decía que cada experiencia se reutiliza ≈ 2 veces. Con minilotes de 32 cada 4 decisiones se muestrean 8 transiciones por decisión, y cada transición permanece 80 000 decisiones en la memoria: 8 · 80 000 / 80 000 = **8 veces** en promedio. Se corrigió en ambas secciones.
4. **Memoria:** se explicita el cálculo de 2,8 GB (35 280 bytes por transición: estado 4 × 84 × 84 + cuadro nuevo 84 × 84).

## Bitácora de dificultades
- Entender por qué los hiperparámetros del artículo de Atari no se pueden copiar tal cual: con 50 veces menos entrenamiento, valores como la sincronización cada 10 000 pasos o una memoria de 1 M de transiciones no aplican, y hubo que justificar cada ajuste.
- Calcular a mano cuántas veces se reutiliza cada transición mostró un error en el texto original (2 en lugar de 8): las cifras «intuitivas» hay que verificarlas.
