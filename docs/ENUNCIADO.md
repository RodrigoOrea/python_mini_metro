# Course Project | A ML app that solves real problems

**Objetivo del curso:** Desarrollar una aplicación que utilice cualquier algoritmo de Machine Learning como capacidad central con potencial real de resolución de problemas.

---

## 1. Requisitos Oficiales del Profesor

- **Algoritmo (`ANY ML algorithm`):** Incluye cualquier algoritmo de Machine Learning, Deep Learning (DL) o LLM. Se recomienda emplear algoritmos vistos en la asignatura, aunque no existe limitación estricta.
- **Flujo completo de ML (`Complete an ML workflow`):** El proyecto debe cubrir de extremo a extremo:
  1. Recolección de datos (*data collection*).
  2. Limpieza y preparación de datos (*data cleaning and preparation*).
  3. Entrenamiento y ajuste de modelos (*model training/tuning*)[cite: 3].
  4. Inferencia del modelo (*model inference*)[cite: 3].
  5. Análisis de resultados (*result analysis*)[cite: 3].
- **Componentes de ingeniería (`Engineering components`):** La aplicación debe contar con una interfaz *frontend* orientada al usuario (por ejemplo, una web sencilla o un entorno interactivo de juego en tiempo real)[cite: 3].
- **Equipo y entrega (`Team & submission`):** Proyecto estrictamente individual[cite: 3]. Antes de finalizar el año se debe entregar el repositorio de código con documentación detallada[cite: 3].
- **Presentación (`Presentation: Week 16-17`):** Presentación corta en clase (diapositivas + demostración en vivo / *live demo*)[cite: 3].
- **Uso de IA (`AI coding`):** Se permite el uso libre de herramientas de programación asistida por IA, bajo la condición indispensable de comprender en profundidad todo el código producido[cite: 3].

---

## 2. Especificación Técnica del Proyecto (Mini Metro ML)

- **Problema a resolver:** Resolución automatizada y visualización en tiempo real del videojuego **Mini Metro** mediante distintos paradigmas de Machine Learning y optimización.
- **Repositorio base:** Fork de `https://github.com/yanfengliu/python_mini_metro.git`.
- **Estado actual / Progreso previo:**
  - Implementado un primer enfoque baseline basado en optimización combinatoria guiada por la función objetivo $H(G)$ dentro de `heuristic_rorea_obj` (este módulo debe conservarse y servir como referencia comparativa).
- **Entregables técnicos esperados:**
  1. **Múltiples enfoques comparables:** Implementar y evaluar distintos métodos (ej. heurística combinatoria $H(G)$ como baseline, aprendizaje supervisado / *Imitation Learning* a partir de trayectorias, y Aprendizaje por Refuerzo / DRL como DQN, PPO o GNN+RL).
  2. **Evaluación comparativa rigurosa:** Analizar el rendimiento de cada método (puntuación media de pasajeros transportados, días sobrevividos, tiempo de inferencia por paso, estabilidad) y detallar cómo se ha implementado cada uno.
  3. **Interfaz visual / Live Demo:** Selector interactivo en el entorno gráfico de `python_mini_metro` que permita elegir el agente/algoritmo en ejecución y visualizar en directo su toma de decisiones para la demo de clase.