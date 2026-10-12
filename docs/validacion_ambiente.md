# Validación del ambiente CarRacing-v3

Registro de la revisión de la sección 2 del README (acciones, observaciones, recompensas y terminación). Los cambios resultantes se integraron en `main` en los commits `7a3ffb1`, `bcd2d37` y `799d2da`.

### Qué revisé
Contrasté la sección 2 del README con la documentación oficial de CarRacing-v3 y con el código fuente del ambiente (`gymnasium/envs/box2d/car_racing.py`, Gymnasium 1.4.0), y ejecuté `scripts/explorar_ambiente.py`.

### Salida de `uv run python scripts/explorar_ambiente.py`
```
== Ambiente original ==
Observación: Box(0, 255, (96, 96, 3), uint8) | dtype uint8
Acciones:    Discrete(5) -> {0: 'nada', 1: 'girar a la derecha', 2: 'girar a la izquierda', 3: 'acelerar', 4: 'frenar'}
Límite de cuadros por episodio: 1000 | umbral de «resuelto»: 900
Recompensa de un paso sin avanzar: -0.1
Baldosas de la pista (semilla 0): 319 -> cada baldosa nueva vale 3.13

== Ambiente preprocesado (lo que ve la red) ==
Observación: Box(0, 255, (4, 84, 84), uint8) | dtype uint8
Forma de un estado: (4, 84, 84) uint8

== Agente aleatorio ==
media -46.00 | desv 18.61 | min -81.30 | max -5.26 | avance medio 5.6 % | 50 episodios
```
La línea base aleatoria se reproduce **idéntica** a `results/linea_base_aleatoria.json`.

### Puntos verificados
- [x] Script ejecutado y salida pegada arriba.
- [x] Tabla de acciones contrastada con la documentación: 1 = derecha y 2 = izquierda es correcto (en el código, la acción 1 aplica la misma dirección que `steer=+1` en la versión continua). Acciones `Discrete(5)`, dtype `int64` verificado.
- [x] Tipo, forma y dtype confirmados: original `Box(0, 255, (96, 96, 3), uint8)`; procesado `(4, 84, 84) uint8`.
- [x] Recorte del tablero validado: el tablero mide 5·H/40 = 12 filas; las filas 0–83 son pista y las filas 84–95 son 100 % negras. El recorte `obs[:84]` quita exactamente el tablero.
- [x] La justificación de apilar 4 cuadros (propiedad de Markov) es clara.

### Cambios realizados
1. **`scripts/explorar_ambiente.py` — corrige la figura de preprocesamiento.** El panel «Original 96×96×3» no correspondía al mismo instante que los 4 cuadros apilados: el script aceleraba durante el zoom inicial (74 cuadros con «acelerar»), mientras que el wrapper salta el zoom con «nada» (50 cuadros) y luego acelera 24. Comprobado: diferencia media de píxeles con el cuadro t‑0 de 11,7 antes y de 0,0 después de la corrección.
2. **`docs/figuras/preprocesamiento.png`** regenerada con el script corregido.
3. **README §2:**
   - Se corrige «único ambiente de control clásico cuya observación es una imagen»: CarRacing pertenece a Box2D, no a control clásico, y los juegos de Atari también entregan imágenes. Ahora dice que es el único de la familia Box2D con observación de imagen.
   - Tabla de acciones con la magnitud real que aplica el simulador (dirección ±0,6, acelerador 0,2, freno 0,8) y su efecto en la conducción.
   - −100 por salir del área de juego: se precisa que reemplaza la recompensa de ese cuadro y el límite (|x| o |y| > 333).
   - Terminación por vuelta completa: no exige todas las baldosas; también termina si el auto vuelve a la baldosa de salida tras visitar más del 95 % (`lap_complete_percent=0.95`).

### Bitácora de dificultades
- La figura parecía correcta a simple vista; solo al comparar los píxeles de ambos paneles se vio que mostraban instantes distintos, por cómo el zoom inicial interactúa con el wrapper.
- La documentación oficial no detalla la regla del 95 % ni la magnitud de las acciones discretas; hubo que leer el código fuente del ambiente para confirmarlas.
