# J3: decisiones (formato del notebook)

Cifras en resultados/j3_resumen.md.

**Modelo final: HGB (no RF).**
- Alternativas: RF (PR-AUC cv 0,5781 ± 0,0130), HGB (0,5765 ± 0,0046), árbol, kNN, SVM.
- Por qué: la diferencia con RF (0,0016) es menor que la desviación entre folds, así que están empatados en validación. El HGB es más estable entre folds, pesa ~3 MB y tiene `decision_function`, que la app usa para verificar la aditividad de SHAP en log-odds; RF no la tiene. kNN, SVM y el árbol quedan claramente por debajo.
- Evidencia: tabla de búsqueda; en 2024 (solo como contexto, no se usó para elegir) HGB 0,584 y RF 0,580.

**Sin categorías nativas (opción b del plan, sección 0 punto 7).**
- Alternativas: (a) nativas + explicador genérico; (b) códigos ordinales + TreeExplainer.
- Por qué: las nativas ganan +0,003 de PR-AUC cv, dentro del ruido entre folds; la app se niega a explicar un HGB con nativas. Con códigos, SHAP es exacto.
- Evidencia: tabla "nativas frente a códigos"; error de aditividad máximo 7e-13 en 3.000 contratos de 2024.

**Ablación de `dias_firma_a_inicio` y `fin_planeado_cruza_anio`: la ventaja no depende de ellas.**
- Sin las dos, la diferencia de PR-AUC contra la regla pasa de +0,114 a +0,110 y la de ROC-AUC dentro de entidad de +0,201 a +0,198; los intervalos siguen excluyendo 0. Quitar las 8 `FEATURES_POR_VERIFICAR` no cambia nada (0,584). Se mantienen; la limitación de la F1 se cita en 3.6 sin cambio de conclusión.

**Artefacto v1 entrenado con 2019-2023 (no se reentrena con 2019-2025 todavía).**
- Por qué: así los percentiles del puntaje (test 2024) y `entidades_vistas` son coherentes con el modelo, y las métricas del .json son fuera de muestra. El reentrenado 2019-2024 no mejora el PR-AUC en 2025 (0,550 frente a 0,550), así que no hay ganancia medida; el reajuste queda al protocolo campeón-retador de G3.

**Notas.**
- Variante C sin `visualizaciones_del`: no aplica, la base v1 no trae esa columna (fuga) y `FEATURES` ya incluye las del proceso. [VERIFICADO en `src/features.py`]
- `el_contrato_puede_ser_prorrogado` (excluida): ablación hecha. Agregarla al HGB final mueve el PR-AUC 2024 en −0,0010 y el ROC-AUC en +0,0005 (dentro del ruido entre folds); confirma dejarla fuera de `FEATURES`. [VERIFICADO: `src/evaluar_j3_extra.py`, `resultados/j3_extra_resumen.md`]
- Contraste "con fuga" de 3.6 (6 montos): no se reconstruye con la base v1 porque esos montos son fuga confirmada y no están en la base (solo en `data/raw/` para el EDA de fuga; el crudo lo tiene Kevin). Se cita el contraste verificado de la F0: al meterle los 6 montos al HGB limpio, ROC-AUC 0,751 → 0,807 y PR-AUC 0,333 → 0,429. [VERIFICADO en la F0; `src/evaluar_j3_extra.py` deja el camino listo si el crudo aparece]
- Umbrales de riesgo (p70 y p90 del puntaje 2024) son provisionales: se cuadran con Geronimo.
- Las tasas de train (0,289) y test 2024 (0,247) difieren: ROC-AUC no cambia por eso, pero el PR-AUC y la probabilidad cruda sí; por eso la app muestra percentil y nivel.
