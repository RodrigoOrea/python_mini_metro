# Guía Técnica de Algoritmos y Arquitectura — `python_mini_metro`

Este documento explica en detalle cómo funciona el entorno de simulación, cómo se selecciona cada acción y cómo evolucionan los siete algoritmos implementados en el proyecto (de la heurística *baseline* hasta el planificador paramétrico optimizado con CEM, la red neuronal residual contextual y el adaptador para el juego real en Unity).

---

## 1. El entorno de decisión: `SemanticMetroEnv` y `event_gate.py`

En lugar de controlar el juego mediante clics y arrastres de píxeles (donde los agentes de RL profundo fallaban al intentar acertar coordenadas exactas), los algoritmos operan sobre `SemanticMetroEnv` (`src/rl/semantic_env.py`), una interfaz discreta basada en grafos.

### Espacio de acciones (`ACTION_TABLE`)
En cada paso de decisión, el entorno calcula una máscara booleana de acciones legales (`env.action_masks()`) sobre las siguientes operaciones:

| Acción (`ActionKind`) | Qué hace en el grafo del metro |
| :--- | :--- |
| `WAIT` (índice `0`) | Deja avanzar la simulación sin modificar líneas ni flota. |
| `PURCHASE_LINE` | Compra una ranura de línea nueva cuando hay créditos suficientes. |
| `CONNECT(u, v)` | Crea una nueva línea conectando dos estaciones `u` y `v` (`u < v`). |
| `EXTEND_LINE(l, s)` | Añade la estación `s` al **final** (`route[-1]`) de la línea `l`. |
| `PREPEND_LINE(l, s)` | Añade la estación `s` al **principio** (`route[0]`) de la línea `l`. |
| `REMOVE_LINE(l)` | Demuele la línea `l` conservando a los pasajeros (recalculan su ruta). |
| `ASSIGN_LOCOMOTIVE(l)` | Asigna una locomotora libre del inventario a la línea `l`. |
| `ATTACH_CARRIAGE(l)` | Acopla un vagón libre del inventario a un tren elegible de la línea `l`. |

### Aceleración por eventos (`src/rl/event_gate.py`)
En una partida típica hay ~`7,989` pasos de decisión brutos, de los cuales más del **99.8 %** son `WAIT` forzados (mientras los trenes circulan entre estaciones). `event_gate.py` avanza automáticamente los pasos `WAIT` hasta que aparece una estación nueva o cambia el conjunto de acciones legales, reduciendo las consultas de `6,859.6` a `50.9` por episodio (~`135×` más rápido) con **0.00** de divergencia en la puntuación.

---

## 2. Evolución de los algoritmos implementados

Todos los resultados comparativos de la tabla resumen corresponden a la evaluación sobre el conjunto de prueba reservado **Test** (semillas `9000..9029`, `n = 30` episodios completos).

| Algoritmo | Módulo principal | Media en Test (`9000..9029`) | Idea clave |
| :--- | :--- | :---: | :--- |
| `heuristic` (*baseline*) | `src/rl/heuristic.py` | **265.60** | Cascada voraz por distancia mínima (1 sola línea). |
| `load_objective` | `src/rl/heuristic_load_objective.py` | **709.63** | Inserción incremental minimizando función objetivo $H(G)$. |
| `multiline_balancing` | `src/rl/heuristic_multiline.py` | **714.40** | Apertura multilínea, reserva de flota y trasbordos (*hubs*). |
| `global_replan` | `src/rl/heuristic_global_replan.py` | **773.77** | Replanificación combinatoria global (2-opt + intercambios). |
| `phase2_rl` (PPO) | `src/rl/contextual_ml.py` | **847.17** | MLP residual contextual ajustado con PPO recortado. |
| `phase2_il` (IL) | `src/rl/contextual_ml.py` | **855.93** | MLP residual contextual preentrenado con Imitation Learning. |
| `parametric_cem` (Mejor) | `src/rl/policy_parametric_cem.py` | **866.20** | Planificador global parametrizado y optimizado con CEM (CRN). |

---

### 2.1. `heuristic` — Heurística *baseline* (`src/rl/heuristic.py`)

Es el punto de partida del proyecto y la barra mínima que cualquier método más avanzado debe superar. Implementa una **cascada de prioridades determinista** en `choose(env)`:

1. **Prioridad 1 — `PURCHASE_LINE`**: si hay una acción legal de compra de línea, la ejecuta de inmediato.
2. **Prioridad 2 — `EXTEND_LINE` / `PREPEND_LINE`**: busca todas las estaciones no servidas (`unserved`) y todos los extremos actuales de las líneas existentes (`route[0]` y `route[-1]`). Calcula la distancia euclídea entre la estación nueva y cada extremo, y ejecuta la extensión o prepending de **menor distancia** (`gap`).
3. **Prioridad 3 — `CONNECT`**: solo si la regla 2 no aplicó (por ejemplo, al empezar la partida sin líneas) y hay $\ge 2$ estaciones sin servicio, conecta el par de estaciones en `unserved` con menor distancia euclídea.
4. **Prioridad 4 — `ASSIGN_LOCOMOTIVE` y `ATTACH_CARRIAGE`**: si hay locomotoras o vagones disponibles para asignar, asigna el primero legal (`ASSIGN_LOCOMOTIVE` antes que `ATTACH_CARRIAGE`).
5. **Prioridad 5 — `WAIT`**: en cualquier otro caso devuelve `0` (`WAIT`).

**Limitación fundamental**: nunca usa `REMOVE_LINE` y, al priorizar `EXTEND_LINE`/`PREPEND_LINE` sobre `CONNECT`, cada estación nueva se engancha a la primera línea. Aunque compre ranuras nuevas, casi siempre acaba con **una única línea sobrecargada** (`lines = 1`), estancándose en **265.60** entregas medias.

---

### 2.2. `multiline_balancing` — Heurística de balanceo multilínea (`src/rl/heuristic_multiline.py`)

Corrige directamente el estancamiento de línea única del *baseline* mediante cinco reglas explícitas:

1. **Compra de líneas**: ejecuta `PURCHASE_LINE` en cuanto es legal.
2. **Prioridad crítica de locomotoras y reasignación dinámica**: si alguna línea activa tiene `0` locomotoras (`uncrewed`), le asigna una de inmediato. Si no quedan locomotoras libres en el inventario, encola una retirada (`queue_locomotive_unassignment`) desde la línea que tenga más locomotoras ($\ge 2$) para mover capacidad hacia la línea vacía.
3. **Arranque de nuevas líneas con estación de trasbordo (*hub*)**: si hay menos líneas creadas que ranuras desbloqueadas (`len(mediator.paths) < mediator.get_unlocked_num_paths()`) y existe alguna estación en `unserved`, ejecuta `CONNECT` conectando esa estación nueva con su vecina más cercana (creando un *hub* de trasbordo si el otro extremo ya estaba servido).
4. **Balanceo por alcance (*catch-up*)**: en `_eligible_lines_for_extension`, si la línea más reciente tiene menos estaciones que la línea anterior, todas las extensiones (`EXTEND_LINE`/`PREPEND_LINE`) se dedican exclusivamente a esa línea nueva hasta igualar su número de estaciones.
5. **Flota equilibrada**: las locomotoras sobrantes se asignan a la línea con menos locomotoras.

**Resultado**: salta de `265.60` a **714.40** entregas medias en Test al aprovechar varias líneas simultáneamente y evitar líneas sin trenes.

---

### 2.3. `load_objective` — Heurística guiada por función objetivo (`src/rl/heuristic_load_objective.py`)

Sustituye las reglas manuales de distancia local por una **función objetivo global sobre el grafo de la red** $H(G)$:

$$h(s) = \sum_{t \in \text{TiposForma}} \min_{u \in V_t} d(s, u)$$
$$H(G) = \sum_{s \in V} h(s) + \text{LoadPenalty}(G)$$

donde:
- $d(s, u)$ es el coste mínimo del camino más corto (calculado con Dijkstra en el grafo multicapa línea-estación) desde la estación $s$ hasta la estación más cercana de tipo $t$:
  - Coste base `1.0` por parada (*hop*).
  - Penalización física de trayecto $\alpha_{\text{dist}} \cdot (\text{distancia\_px} / d_{\text{ref}})$, con $\alpha_{\text{dist}} = 1.5$ y $d_{\text{ref}} = 250.0$.
  - Penalización $\beta_{\text{transfer}} = 1.5$ por cada trasbordo entre líneas en una estación compartida.
  - Penalización `unreach_penalty = 30.0` si una forma es inalcanzable desde $s$.
- $\text{LoadPenalty}(G)$ penaliza rutas excesivamente largas por locomotora asignada.

En cada decisión, simula el grafo resultante $G_a$ de cada acción candidata (`CONNECT`, `EXTEND_LINE`, `PREPEND_LINE`) y elige la que **minimiza** $H(G_a)$. Alcanza **709.63** entregas en Test sin depender de heurísticas de orden ad-hoc.

---

### 2.4. `global_replan` — Replanificación combinatoria global (`src/rl/heuristic_global_replan.py`)

El análisis empírico mostró que la inserción puramente incremental produce rutas un ~34 % más largas que el orden óptimo de las mismas estaciones. `global_replan` resuelve el **Transit Network Design Problem (TNDP)** completo sobre todas las estaciones:

- **Función objetivo regularizada**:
  $$H_{\text{global}}(G) = H(G) + \text{BalancePenalty}(G)$$
  donde $\text{BalancePenalty}(G)$ suma $\text{balance\_weight} \cdot \left((\text{norm\_len})^2 + 2 \cdot (\text{norm\_stops})^2\right)$ por línea ($\text{balance\_weight} = 1.8$), penalizando de forma convexa que una línea acapare demasiadas paradas o distancia frente a otras.
- **Búsqueda combinatoria**:
  - **Intra-línea**: operadores *2-opt* (invertir subsegmentos para eliminar cruces) y *relocate* (mover una estación a otra posición de la misma línea).
  - **Inter-línea**: transferencia de estaciones entre líneas distintas e intercambio (*swap*) de estaciones entre pares de líneas.
- **Ejecución segura con histéresis**: cuando un rediseño ahorra al menos un umbral mínimo absoluto y relativo (con *cooldown* de `500` decisiones), reconstruye **una sola línea cada vez** mediante una secuencia `RebuildPlan`:
  `REMOVE_LINE` $\rightarrow$ `CONNECT` $\rightarrow$ `EXTEND_LINE` / `PREPEND_LINE`.

**Resultado**: sube a **773.77** entregas medias en Test.

---

### 2.5. `parametric_cem` — Planificador paramétrico optimizado con CEM (`src/rl/policy_parametric_cem.py`, `src/rl/objective_ml.py`, `src/rl/replan_ml.py`, `scripts/train_parametric_cem.py`)

Es el algoritmo con **mayor rendimiento global (866.20 en Test, máximo 1508)**. Parte de `global_replan` (garantizando `0.00` de deriva en los parámetros ancla `ANCHOR_PARAMS`), pero parametriza sus decisiones críticas con un vector de 15 hiperparámetros (`ParametricPolicyParams`) entrenados directamente sobre entregas finales:

1. **Demanda en vivo (`LiveDemandSnapshot`)**: incorpora el tamaño de las colas y el cuadrado del tiempo de espera de los pasajeros en cada estación, priorizando conectar o reforzar estaciones con presión real.
2. **Riesgo continuo de demolición**: antes de ejecutar `REMOVE_LINE`, evalúa el riesgo de vaciar temporalmente una línea con pasajeros esperando cerca del límite de sobrecarga (40 s); si el riesgo supera el umbral aprendido, veta reconstrucciones discrecionales.
3. **Asignación y rebalanceo *minimax* de flota**: asigna locomotoras y vagones según la presión efectiva de cada línea ($\text{presión} = \text{demanda} / \text{unidades\_efectivas}$, donde cada vagón aporta `0.55` unidades), y traslada locomotoras sin vagones desde la línea más holgada a la más saturada cuando la diferencia supera el umbral.
4. **Entrenamiento con Cross-Entropy Method (CEM) y CRN**: `scripts/train_parametric_cem.py` evalúa candidatos usando **Common Random Numbers (CRN)** emparejados (mismas semillas `1000..1009`) y actualiza la media y desviación hacia los candidatos élite en Google Cloud VM.

**Resultado**: **866.20** entregas medias en Test (`+92.43 ± 96.09` sobre `global_replan`, con balance victorias/derrotas/empates `19/9/2`).

---

### 2.6. `contextual_residual` — Política neuronal residual contextual (`src/rl/contextual_ml.py`, `scripts/train_contextual_il.py`, `scripts/train_contextual_rl.py`)

En lugar de usar un único vector fijo `BEST_PHASE1_PARAMS` durante toda la partida, la Fase 2 introduce una red neuronal MLP residual en NumPy puro (`ContextualResidualMLP`, con una capa oculta `tanh`) que ajusta los 15 parámetros en función del estado instantáneo del tablero:

- **Vector de contexto (`16` características normalizadas)**: número de estaciones, líneas, media y máximo de paradas, fracción sin servicio, créditos, locomotoras y vagones libres, total de metros/vagones, espera máxima, percentil 90 de espera al cuadrado, llenado medio de estaciones, carga máxima de línea, desviación estándar de carga y desequilibrio entre líneas.
- **Inicialización a cero (*zero-init*)**: los pesos de la capa de salida arrancan en `0`, garantizando que sin entrenamiento adicional la política coincide bit a bit con `parametric_cem`.
- **Entrenamiento en dos etapas**:
  1. **Imitation Learning (`phase2_il`, `scripts/train_contextual_il.py`)**: preentrenamiento supervisado en PyTorch sobre un dataset contrafactual CRN de ventajas por dirección (`scripts/collect_phase2_dataset.py`, semillas `1000..1027`). Alcanza **855.93** en Test.
  2. **Reinforcement Learning (`phase2_rl`, `scripts/train_contextual_rl.py`)**: ajuste fino con **PPO recortado** (*clipped PPO*) en PyTorch. Alcanza **847.17** en Test (`9/8/13` frente a `parametric_cem`).

---

### 2.7. `open_topology_planner` — Planificador para el juego real (`src/rl/open_topology_planner.py`, `src/rl/real_game_adapter.py`)

Adapta la lógica de `parametric_cem` al juego comercial de Mini Metro en Unity mediante `OpenMetroView` y el protocolo `bridge-rpc-v1`:

- **Sin mutación del entorno**: a diferencia del simulador en Python, no puede clonar o avanzar un `Mediator` en memoria; planifica sobre instantáneas inmutables (`OpenStation`, `OpenLine`) recibidas por TCP NDJSON.
- **Topología abierta y conciencia de túneles**: verifica el presupuesto de cruces de agua (`crossing_cost_fn`, `tunnels_available`) antes de proponer `CONNECT`, `INSERT_STATION` o `REPLACE_LINE`, permitiendo inserciones con coste neto de túneles igual a cero cuando no quedan túneles libres.
- **Acciones exclusivas del juego real**: soporta `INSERT_STATION` (insertar una estación intermedia sin demoler la línea completa), `RELOCATE_LOCO` (reubicar locomotoras), `PLACE_INTERCHANGE` (colocar intercambiadores en *hubs* congestionados) y `PICK_UPGRADE` (elegir automáticamente las mejoras semanales en `NewAssetScreen`).
