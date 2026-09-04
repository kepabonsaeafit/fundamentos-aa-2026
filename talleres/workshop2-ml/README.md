# Workshop 2 - Machine Learning (Unidad 2)

Pipelines, entrenamiento, comparacion de modelos y validacion cruzada sobre un problema de
regresion y uno de clasificacion.

**Datasets asignados:**
- Regresion - Flight Price Prediction: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction
- Clasificacion - Thyroid Disease Data: https://www.kaggle.com/datasets/jainaru/thyroid-disease-data/data

## Equipo
- Kevin Pabon Nino - @kepabonsaeafit
- Jose Alejandro Jimenez - @ll333ll
- Geronimo Montes Acebedo - @gmonsteaa

## Notebooks

| Notebook | Problema | Target |
| --- | --- | --- |
| `workshop2_regresion_vuelos.ipynb` | Regresion | `price` (rupias indias) |
| `workshop2_clasificacion_tiroides.ipynb` | Clasificacion binaria | `Recurred` (Yes/No) |

Los dos notebooks estan ejecutados: las tablas y las graficas se ven en GitHub sin correr nada.

## Datos

Los CSV no se versionan (misma regla que el workshop 1). Para descargarlos:

```bash
pip install kaggle
kaggle datasets download -d shubhambathwal/flight-price-prediction -p data/vuelos --unzip
kaggle datasets download -d jainaru/thyroid-disease-data -p data/tiroides --unzip
```

Los notebooks usan `data/vuelos/Clean_Dataset.csv` y `data/tiroides/Thyroid_Diff.csv`, relativos a
esta carpeta.

## Checklist (basado en el enunciado)

### Fase 1 - EDA, limpieza y preprocesamiento (ambos datasets)
- [x] 1. Documentacion del dataset, problema de negocio/salud y variable objetivo
- [x] 2. Carga con pandas, head()/tail(), .shape, .info() y diccionario de datos
- [x] 3. .describe() con interpretacion de 3+ estadisticos por dataset
- [x] 4. Nulos por columna + porcentaje + estrategia de imputacion justificada
- [x] 5. Duplicados identificados y eliminados (reportando cuantos)
- [x] 6. Inconsistencias de formato en categoricas revisadas y corregidas
- [x] 7. Outliers via IQR en columnas numericas, con decision justificada
- [x] 8. Graficas de EDA (barras, pie, histograma, scatter, boxplot) con interpretacion y recomendacion
- [x] 9. Tecnica de codificacion justificada por variable categorica
- [x] 10. Escalamiento de numericas: si aplica y con que tecnica, justificado

### Fase 2 - Division de datos y Pipelines
- [x] 11. Split train / validation / test (70/15/15) con explicacion del proposito de cada conjunto
- [x] 12. Pipeline con ColumnTransformer (numerico y categorico) por dataset
- [x] 13. Verificacion escrita de ausencia de data leakage (.fit() solo sobre X_train)

### Fase 3 - Modelado: Regresion
- [x] 14. Entrenar Regresion Lineal, KNN, Decision Tree, Random Forest y Gradient Boosting
- [x] 15. Predicciones sobre el conjunto de validacion

### Fase 4 - Modelado: Clasificacion
- [x] 16. Entrenar KNN, Decision Tree, Random Forest y Gradient Boosting
- [x] 17. Predicciones sobre el conjunto de validacion

### Fase 5 - Metricas en train y validacion
- [ ] 18. Regresion: MAE, MSE y R2 en train y en val, con analisis de over/underfitting
- [ ] 19. Clasificacion: Accuracy, Precision, Recall, F1 y matriz de confusion en train y en val
- [ ] 20. Tabla comparativa (train + val) para cada problema
- [ ] 21. Seleccion del mejor modelo por problema, con trade-offs

### Fase 6 - Evaluacion final en Test
- [ ] 22. Evaluar los 5 modelos de regresion y los 4 de clasificacion sobre test
- [ ] 23. Tabla final en test y comparacion contra la de validacion

### Fase 7 - Cross Validation
- [ ] 24. Explicacion de K-Fold Cross Validation con palabras propias
- [ ] 25. K-Fold sobre el mejor modelo de regresion
- [ ] 26. K-Fold sobre el mejor modelo de clasificacion
- [ ] 27. Comparacion contra el desempeno en validacion y analisis de estabilidad

### Fase 8 - Conclusiones
- [ ] 28. Parrafo de conclusiones comparando ambos problemas + subplot (2,2) por dataset

## Reparto

| Fase | Responsable |
| --- | --- |
| 1 (puntos 1-10, ambos datasets) | Jose Alejandro Jimenez |
| 2, 3 y 4 (puntos 11-17, ambos datasets) | Kevin Pabon Nino |
| 5 a 8 (puntos 18-28) | Geronimo Montes Acebedo |

La Fase 1 deja definida al final de cada notebook (puntos 9 y 10) la especificacion de
preprocesamiento -listas de columnas numericas, nominales y ordinales con sus ordenes de
categorias- que es justo lo que consume el ColumnTransformer de la Fase 2.

## Handoff a la Fase 5

Las Fases 2-4 dejan los modelos ya entrenados y las predicciones ya generadas, ejecutadas sobre
los datasets reales completos. La Fase 5 no necesita reentrenar nada: solo calcular metricas
sobre estas estructuras.

**En `workshop2_regresion_vuelos.ipynb`:**

| Objeto | Contenido |
| --- | --- |
| `X_train`/`X_val`/`X_test`, `y_train`/`y_val`/`y_test` | Particion 70/15/15 (210.107 / 45.023 / 45.023 filas) |
| `preprocesador` | ColumnTransformer ajustado solo sobre X_train (34 columnas) |
| `pipelines_reg` | `{modelo: Pipeline}` - los 5 modelos entrenados |
| `predicciones_reg` | `{modelo: {"train": array, "val": array}}` |
| `tiempos_reg` | `{modelo: segundos}` - insumo del trade-off del punto 21 |

**En `workshop2_clasificacion_tiroides.ipynb`:**

| Objeto | Contenido |
| --- | --- |
| `X_train`/`X_val`/`X_test`, `y_train`/`y_val`/`y_test` | Particion 70/15/15 estratificada (254 / 55 / 55 pacientes); target 0/1, **1 = Yes = clase positiva** |
| `construir_preprocesador(incluir_response)` | Fabrica del ColumnTransformer; la Fase 7 la necesita para el KFold |
| `prep_seguimiento` (32 features) / `prep_diagnostico` (31) | Los dos escenarios: con y sin `Response` |
| `pipelines_clf` | `{(escenario, modelo, variante): Pipeline}` - 14 pipelines |
| `predicciones_clf` | `{clave: {"train", "val", "proba_val"}}` |
| `tiempos_clf`, `resumen_clf` | Tiempos de entrenamiento y tabla en formato largo |

Tres cosas que hay que respetar al continuar:

1. **`X_test` y `y_test` no se tocan hasta la Fase 6.** Estan definidos desde la Fase 2 solo
   porque la particion se hace de una vez.
2. **En clasificacion la metrica principal no es accuracy**, sino recall o F1 sobre la clase
   positiva: un modelo que respondiera siempre "No recurre" ya acertaria el 70 %.
3. **Los dos escenarios de tiroides se comparan entre si, no contra el otro.** El modelo de
   diagnostico (sin `Response`) va a rendir peor y eso no lo hace peor modelo: responde una
   pregunta distinta, la que se puede contestar el dia del diagnostico.

Ambos notebooks ya estan ejecutados de punta a punta con los datasets reales completos
(300.153 filas de vuelos, 364 pacientes de tiroides): las tablas, graficas y numeros que se ven
en GitHub corresponden a una corrida real, no a un esqueleto sin ejecutar.

## Entregable
Dos notebooks .ipynb en este repo, uno de regresion y uno de clasificacion. Solo se sube el link
del repo a la plataforma.

**Nota:** sustentacion aleatoria -> (Sustentacion+Entrega)/2
