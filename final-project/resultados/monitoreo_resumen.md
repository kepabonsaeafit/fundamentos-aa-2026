# Simulacion de monitoreo 2024-2025

Modelo congelado: entrenado con 2019-2023 (contratos maduros, etiqueta `y`), sin reentrenar. Base v1, corte 2026-10-26.
Se generan con `python -m src.monitoreo`. Reglas y justificacion en `docs/REAJUSTE.md`.

## Entradas y puntaje (mensual, todos los contratos)
- Entradas frente al mismo mes de 2022-2023: 16 meses normales y 8 para revisar (2 o mas de las 10 variables con PSI >= 0.1, o alguna >= 0.25).
- Meses para revisar: 2024-01, 2024-03, 2024-04, 2024-06, 2024-09, 2025-01, 2025-02, 2025-06.
- Puntaje frente al mismo mes del anio anterior (2025): 12 meses normales, 0 en aviso, 0 en alarma. PSI maximo: 0.094.
- Contexto: entre anios consecutivos del entrenamiento (2021-2023) el PSI de las entradas ya tenia mediana 0.090 y el 29% de los valores superaba 0.25; en 2024-2025 la mediana es 0.032 y el 1% supera 0.25. El PSI avisa de un cambio, pero no dice si el modelo se degrado: por eso no dispara el reentrenamiento por si solo.

## Desempeno y etiqueta (trimestral, solo cohortes completas)
- Trimestres completos (>= 95% maduros): 2024Q1, 2024Q2, 2024Q3, 2024Q4, 2025Q1, 2025Q2, 2025Q3.
- Trimestres incompletos, sin evaluar con etiquetas: 2025Q4.
- Aviso por cambio de la tasa de `y` (>= 5% frente al mismo trimestre): 2024Q1, 2024Q2.
- Aviso por cambio de la tasa de `y_dias` (>= 5%): 2025Q1, 2025Q2, 2025Q3.
- Top 1% peor que un contrato al azar (precision < tasa base): 2025Q3.
- Top 1% dominado por una sola entidad (> 50%), informativo: 2024Q2, 2024Q4, 2025Q3.
- Alarma de desempeno (ROC-AUC cae >= 0.03 frente al mismo trimestre, con IC 95 % que excluye 0, 2 trimestres seguidos): ninguno.

## Campeon frente a retador (2019-2023 frente a 2019-2024, evaluados en 2025)
- Contratos de evaluacion: 80.652 (2025Q1, 2025Q2, 2025Q3).
- PR-AUC: campeon 0.544, retador 0.542; diferencia -0.002 [-0.020, +0.032].
- ROC-AUC: campeon 0.746, retador 0.764; diferencia +0.018 [+0.008, +0.028].
- Decision segun la regla de promocion: **no promover: sin evidencia suficiente de mejora**.
