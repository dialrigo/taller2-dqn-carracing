# Validación de los resultados (Rol 6)

Registro de la revisión de la sección 6 del README. Responsable: Andrés Gonzalo Díaz Lázaro.

## Qué revisé
- Cada cifra de §6 contra los archivos de `results/`, recalculándola con pandas (no a ojo).
- Las cuatro figuras de `docs/figuras/` que usa §6: título, ejes, unidades y leyendas.
- Que las conclusiones de §6.4 no afirmen más de lo que muestran los datos.

## Fuente de cada cifra

| Cifra del README | Valor | Archivo y campo |
|---|---|---|
| Decisiones, episodios, duración | 250 000 · 1 186 · 214,7 min (≈ 3,6 h) | `results/dqn_carracing/resumen_entrenamiento.json` (`episodios`, `duracion_min`); `episodios.csv` (último `paso`) |
| Tabla de evaluaciones periódicas (§6.1) | 10 filas, −73,7 … −95,0 y % de pista | `results/dqn_carracing/evaluaciones.csv` (`media`, `avance_pista`) |
| Fase 1: media móvil entre −42 y 0 hasta ≈ 58 000; ≈ 50 a las 80 000 | mín −42,4; cruza 0 en la decisión 58 183; 50,8 en la 80 000 | `episodios.csv`, `retorno` con media móvil de 30 (igual que `plots.py`) |
| Fase 1: 61–81 % «sin progreso» | tramos de 10 000 entre 10 000 y 80 000 | `episodios.csv`, `fin` |
| Fase 2: media móvil ≈ 50 → ≈ 580; «sin progreso» ≈ 64 % → ≈ 5 % | 49,6 → 582,7; 64 % (tramo 80–90 k) → 5 % (tramo 120–130 k) | `episodios.csv` |
| Mejor episodio de entrenamiento | 917,6 en la decisión 131 158, vuelta completa | `episodios.csv` (máximo de `retorno`) |
| Fase 3: media móvil ≈ 310–605; 93 % por tiempo; 0,6 % fuera del mapa | 310,2 – 604,8; 92,8 %; 0,6 % | `episodios.csv` |
| Pérdida ≈ 0,2 → máx. ≈ 2,9; Q medio ≈ 0 → ≈ 104 | 0,18 → 2,87; −0,3 → 104,4 | `episodios.csv` (`perdida`, `q_medio`, media móvil de 30) |
| Tabla de evaluación final (§6.2) | todas las celdas | `results/evaluacion_final.json` (`mejor_eps0`, `mejor_eps0.05`, `final_eps0`, `final_eps0.05`) |
| Agente aleatorio | −46,0 ± 18,6; −81,3 – −5,3; 5,6 % | `results/linea_base_aleatoria.json` |
| Ajuste lineal retorno vs. avance | pendiente 1 001,7; intercepto −102,6; R² = 0,9998 | `results/evaluacion_episodios.csv` (modelo `mejor`, ε = 0) |
| 48 de 50 por tiempo; 2 vueltas completas | 48 / 2 | `evaluacion_episodios.csv`, `fin` |
| Episodio de la pista 1000 | 816,3 | `evaluacion_episodios.csv`, semilla 1000 |
| Distribución de acciones del mejor modelo | acelerar 62,2 %; izquierda 22,5 %; derecha 13,7 %; nada 1,4 %; frenar 0,3 % | `evaluacion_final.json`, `mejor_eps0.distribucion_acciones` |
| Segunda corrida completa (§6.4, punto 4) | 554 ± 210, mejor punto de control en la decisión 125 000 | descripción del *pull request* #4 (Rol 5); los archivos de esa corrida no están en el repositorio |

## Qué corregí en §6

1. **Fase 1:** decía que el retorno se mantenía entre −40 y 0 hasta la decisión 80 000; la media móvil sale de esa franja hacia la 58 000 y llega a ≈ 50. El rango de «sin progreso» pasa de «60–80 %» a «61–81 %», con el tramo explícito.
2. **Fase 2:** la media móvil arranca en ≈ 50, no en ≈ 0, y llega a ≈ 580; se precisa la caída de «sin progreso» (≈ 64 % → ≈ 5 %).
3. **Fase 3:** el rango real de la media móvil es ≈ 310–605 (no 350–600); se agrega el 93 % por tiempo.
4. **Pérdida y Q:** máximo de la pérdida ≈ 2,9; Q medio final ≈ 104.
5. **§6.3:** girar a la izquierda es 22 % (22,46 %), no 23 %.
6. **§6.2:** se agrega el R² del ajuste lineal y la comparación 655,6 (5 pistas de selección) frente a 665,0 (50 pistas nuevas), que descarta que la elección del mejor punto de control haya inflado el resultado.
7. **§6.4, punto 2:** «no por errores de trayectoria» era demasiado fuerte, porque §6.3 muestra salidas del borde de la pista; queda «principalmente por velocidad», con la evidencia.
8. **§6.4, punto 4:** se incorpora la reproducción completa del Rol 5 (554 ± 210 con la misma semilla en otro equipo). La afirmación «una sola semilla» era correcta, pero ahora hay evidencia de que la misma semilla no reproduce el resultado entre máquinas.

## Figuras

| Figura | Estado | Cambio |
|---|---|---|
| `curva_aprendizaje.png` | Correcta: título, ejes, unidades y leyenda | — |
| `fin_episodios_y_perdida.png` | El eje x del panel izquierdo marcaba 25 000 para el tramo 0–25 000 y los ejes del derecho no decían que son medias móviles | Etiquetas aclaradas en `plots.py` |
| `evaluacion_final.png` | Las líneas de 900 y del agente aleatorio del panel izquierdo no tenían leyenda | Leyenda agregada en `plots.py` |
| `politica_episodio.png` | Correcta | — |
