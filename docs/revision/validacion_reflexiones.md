# Validación de las reflexiones y control de calidad (Rol 7)

Registro de la revisión de las secciones 7 y 8 del README y del control de calidad final. Responsable: Andrés Gonzalo Díaz Lázaro.

## Qué revisé
- Cada afirmación cuantitativa de §7 contra los modelos entregados (`models/`) y los archivos de `results/`.
- Las bitácoras de dificultades de los *pull requests* #1 a #4 y los registros `validacion_ambiente.md` y `validacion_red.md` de esta carpeta, para reescribir §8.
- El README completo, buscando contradicciones entre secciones.
- Las tablas de integrantes del README y de los documentos internos de coordinación del equipo (retirados del repositorio en la auditoría final).

## Salida de `uv run python scripts/diagnostico_colapso.py`
```
== 1. Estado inicial (auto detenido) ==
mejor  pista 1000: nada 59.5, girar a la derecha 58.4, girar a la izquierda 58.6, acelerar 61.7, frenar 56.2
       acción elegida en 50 pistas: {'acelerar': 50} | brecha media entre la 1.ª y la 2.ª acción: 1.70
final  pista 1000: nada 91.3, girar a la derecha 93.1, girar a la izquierda 93.4, acelerar 91.7, frenar 89.1
       acción elegida en 50 pistas: {'girar a la izquierda': 26, 'girar a la derecha': 24} | brecha media entre la 1.ª y la 2.ª acción: 0.18

== 2. Modelo final, greedy desde el estado inicial ==
decisión 0: girar a la izquierda   recompensa -0.4 | diferencia media de píxeles con el cuadro inicial: 0.01
decisión 1: girar a la izquierda   recompensa -0.4 | diferencia media de píxeles con el cuadro inicial: 0.01
decisión 2: girar a la izquierda   recompensa -0.4 | diferencia media de píxeles con el cuadro inicial: 0.01
decisión 3: girar a la izquierda   recompensa -0.4 | diferencia media de píxeles con el cuadro inicial: 0.01
decisión 4: girar a la izquierda   recompensa -0.4 | diferencia media de píxeles con el cuadro inicial: 0.01
decisión 5: girar a la izquierda   recompensa -0.4 | diferencia media de píxeles con el cuadro inicial: 0.01

== 3. Arranque forzado: 5 decisiones de «acelerar» y luego greedy (20 pistas) ==
mejor  retorno 615.5 ± 202.1 (mín 273.9, máx 902.3)
final  retorno 106.9 ± 24.6 (mín 61.3, máx 148.2)
```

## Qué corregí en §7
1. **Valores Q del estado inicial.** El borrador citaba 49,2 frente a 47,5 (mejor modelo) y 88,0 frente a 86,2 (modelo final); no coinciden con los modelos entregados en ninguna pista de evaluación. Se reemplazan por los valores medidos y por la estadística en las 50 pistas: el mejor modelo elige «acelerar» en 50 de 50 (ventaja media 1,7) y el final elige girar en 50 de 50 (ventaja media 0,18).
2. **«El modelo final no es peor conduciendo» era falso.** Con 5 decisiones forzadas de «acelerar», el modelo final obtiene 107 ± 25 frente a 615 ± 202 del mejor en las mismas 20 pistas. El colapso son dos fallas: la trampa del estado inicial (explica el −95) y una degradación general de la política.
3. **Crecimiento de Q.** El borrador decía «de ≈ 50 a ≈ 100» entre las decisiones 150 000 y 250 000; el Q medio de los minilotes va de ≈ 20 a ≈ 104. Además, el borrador lo presentaba como señal de sobreestimación; con ≈ 2,8 puntos por decisión y γ = 0,99 el valor alcanzable es ≈ 280, así que un Q de 104 no prueba sobreestimación. Se reformula con esa cautela.
4. **Presupuesto.** «Las implementaciones de referencia suelen entrenar de 2 a 5 M» no tenía fuente; se reemplaza por la comparación verificable con los 50 M de cuadros de Mnih et al. (2015).
5. Se incorpora la segunda corrida completa del Rol 5 (554 ± 210) como evidencia de inestabilidad entre corridas.
6. «Qué haríamos distinto», punto 2: el arranque forzado elimina el −95, pero no recupera el modelo final.

## §8
Se eliminó la nota de plantilla y se reescribió con las dificultades reales del grupo, agrupadas en técnicas, conceptuales y de verificación, citando el rol que las registró.

## Control de calidad
- §5.2 decía que el presupuesto era *la* razón de no llegar a 900, mientras §7 da tres factores; se matizó a «una de las razones».
- Resumen inicial del README actualizado con el hallazgo del colapso.
- Tablas de integrantes completas.
- Nuevo script `scripts/diagnostico_colapso.py` agregado a la sección 1 y al árbol del repositorio.
- Cifras verificadas sin cambios: ε ≈ 0,905 al terminar el llenado; 35 280 bytes por transición y ≈ 2,8 GB; ≈ 338 episodios en memoria; horizonte de 400 cuadros ≈ 8 s; 40 % del entrenamiento en exploración.

## Bitácora de dificultades
- Las cifras del borrador de §7 sonaban razonables y eran específicas, pero no salían de los modelos entregados; hubo que montar un entorno, cargar los dos modelos y medir.
- La explicación del colapso era convincente y estaba incompleta: solo un experimento dirigido (arranque forzado) mostró que el modelo final también conduce peor.
- Calcular el valor descontado alcanzable obligó a retirar una conclusión (sobreestimación) del primer borrador de esta misma revisión.
