# Guía del equipo — Grupo 10

Esta guía explica cómo participa cada integrante en el Taller 2. El objetivo es que **cada persona revise, entienda y mejore una parte concreta**, y que su aporte quede registrado en GitHub como un *pull request* (PR), como sugirió el profesor en la Sesión 5 («pueden hacer un *pull request* para mostrar sus aportes al proyecto»).

**Fecha límite de integración de los PR:** sábado 10 de octubre, 8:00 p. m.
**Revisión final (Rol 7) y entrega en Moodle:** domingo 11 y lunes 12 de octubre (cierre 23:59).

---

## 1. Roles

| Rol | Responsable | Usuario GitHub | Entregable (su PR) | Tiempo estimado |
|---|---|---|---|---|
| 1 · Integración | Diego Ríos | `dialrigo` | Repositorio base, revisión y fusión de PRs, entrega | — |
| 2 · Red neuronal e hiperparámetros | Nataly Valbuena | _por definir_ | Verificación de la arquitectura y justificación de hiperparámetros (README §5) | 1–2 h |
| 3 · Ambiente | _Integrante 3_ | | Exploración del ambiente y validación de acciones, observaciones y recompensas (README §2) | 1–2 h |
| 4 · Flujo de entrenamiento | _Integrante 4_ | | **Diagrama propio dibujado a mano** del ciclo DQN y particularidades (README §3–4) | 2 h |
| 5 · Experimentos | _Integrante 5_ | | Reproducción desde cero, corrida corta y captura de MLflow (README §1) | 1–2 h |
| 6 · Resultados | _Integrante 6_ | | Verificación de cifras y figuras (README §6) | 1–2 h |
| 7 · Reflexiones y QA | _Integrante 7_ | | Reflexión de resultados y de «lo que más costó» con las dificultades reales del grupo; checklist final (README §7–8) | 2 h |

> Escriban su nombre y usuario de GitHub en esta tabla dentro de su propio PR.

El detalle de qué verificar en cada rol está en [`CHECKLIST_REVISION.md`](CHECKLIST_REVISION.md).

---

## 2. Preparación (una sola vez, 15 minutos)

1. **Cuenta de GitHub.** Si no tienen, créenla en https://github.com/signup. Envíen su usuario a Diego para que los agregue como colaboradores (Settings → Collaborators). Acepten la invitación que llega al correo.
2. **Git.** Windows: instalar desde https://git-scm.com/download/win (opciones por defecto). Mac: `xcode-select --install`.
3. **uv** (gestor de entornos que usa el profesor):
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - Mac/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
4. **Clonar e instalar:**
   ```bash
   git clone https://github.com/dialrigo/taller2-dqn-carracing.git
   cd taller2-dqn-carracing
   uv sync
   ```
   Si `uv sync` falla al instalar Box2D en Windows, instalar primero las *Microsoft C++ Build Tools* o usar el notebook de Colab (`notebooks/colab_carracing_dqn.ipynb`), que no requiere instalar nada.
5. **Probar que todo funciona** (1 minuto):
   ```bash
   uv run python scripts/resumen_red.py
   ```

> **Alternativa sin instalar nada:** abrir el notebook en Google Colab (botón en el README) y trabajar ahí. Para el PR pueden editar los archivos directamente en la web de GitHub (paso 3, opción B).

---

## 3. Flujo de trabajo para su aporte

### Opción A — con Git (recomendada)
```bash
git checkout main && git pull                       # partir de la versión más reciente
git checkout -b rol3-ambiente                       # una rama por rol: rol2-red, rol3-ambiente, ...
# ... revisar, ejecutar, editar los archivos de su rol ...
git add README.md docs/figuras/...                  # solo los archivos que cambiaron a propósito
git commit -m "Rol 3: valida espacios y recompensas contra la documentación de Gymnasium"
git push -u origin rol3-ambiente
```
Luego en GitHub: botón **Compare & pull request** → completar la plantilla → **Create pull request** y asignar a Diego como revisor.

### Opción B — desde la web de GitHub
1. Abrir el archivo (por ejemplo `README.md`) → ícono del lápiz (*Edit*).
2. Hacer los cambios.
3. **Commit changes…** → elegir *Create a new branch for this commit and start a pull request* → nombre `rolN-tema`.
4. Completar la plantilla del PR.

### Reglas para que la entrega quede limpia
- Un PR por rol, con título `Rol N · tema`.
- Mensajes de *commit* en español, en infinitivo o presente, que digan qué cambia.
- No subir: `mlflow.db`, `mlruns/`, `checkpoints/`, carpetas `results/prueba_*`, archivos de su computador.
- No modificar el código de otro rol sin comentarlo en su PR.
- Si encuentran un error, abran un *Issue* (pestaña Issues) en lugar de arreglarlo en silencio.

---

## 4. Qué debe quedar demostrado al final

Cada integrante debe poder explicar, si el profesor pregunta:
- qué ve el agente (4 cuadros de 84×84 en gris) y por qué;
- qué hace cada componente de DQN (memoria, ε-greedy, red objetivo, Bellman);
- qué resultado obtuvo el agente y qué limita su desempeño;
- qué revisó él o ella específicamente y qué cambió.

## 5. Entrega en Moodle

La actividad aparece como individual en el aula virtual, por lo que **cada integrante sube el mismo enlace** del repositorio: Simulación y Aprendizaje por Refuerzo → Unidad 3 → *Taller 2 — Resolver un ambiente nuevo de Gymnasium con DQN* → Agregar entrega → pegar `https://github.com/dialrigo/taller2-dqn-carracing` → Guardar cambios. Si el profesor confirma que basta con una entrega por grupo, la hace Diego.
