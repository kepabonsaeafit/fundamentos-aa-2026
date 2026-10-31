# Estrategia de reajuste del modelo (requisito 3.9)

> Leyenda: **[VERIFICADO]** visto en los datos o el código · **[SUPUESTO]** · **[ESTIMADO]** · **[NO EVALUADO]**.
> Todas las cifras salen de `python -m src.monitoreo` sobre la base v1 (corte del 26 de octubre de 2026) con el **modelo v1** (HistGradientBoosting de Jose) reentrenado con 2019-2023. Si el modelo cambia se repite el mismo comando y se actualizan las cifras.

El enunciado pide tres cosas: cómo se detectaría que el desempeño se degradó o que los datos cambiaron, con qué criterio y frecuencia se reentrenaría, y cómo se validaría y pondría en producción la versión nueva. Este documento responde las tres y lo respalda con una simulación de 2024-2025.

## 1. Qué puede cambiar, con ejemplos de este proyecto

| Tipo de cambio | Ejemplo real en los datos [VERIFICADO] |
|---|---|
| **Cambian las entradas** | `dias_firma_a_inicio`: el 48 % de los contratos de 2019 tenía el inicio *antes* de la firma, el 12 % en 2020 y desde 2021 casi 0 %. Por eso la referencia del monitor son 2022-2023 y no todo el entrenamiento: con 2019-2023 el PSI de esta variable llegaba a 0,9 en febrero y marzo. |
| **Cambia el comportamiento** | En 2025-T3 el 82 % del 1 % de mayor puntaje eran contratos de una sola entidad (la Defensoría) y solo el 15 % se amplió, frente al 22 % de un contrato cualquiera. El ROC-AUC global casi no se movió. |
| **Cambia el registro de la etiqueta** | `y_dias` pasa de ~12 % a ~20 % en 2025, lo que es consistente con un cambio en el registro de `dias_adicionados` (no verificado en v1; guardaría solo los días de la última modificación publicada). Con la etiqueta aprobada `y` la tasa casi no se mueve (24,7 % → 27,1 %), así que no apunta a un cambio de comportamiento. |
| **Estacionalidad** | La tasa de `y` va de 34 % en enero a 16 % en octubre y noviembre; el ROC-AUC mensual oscila entre 0,71 y 0,87 solo por la composición de cada mes. |
| **Etiquetas tardías** | El resultado se conoce meses después de firmar. Un contrato sin duración planeada solo madura al año; si se miden solo los maduros, los meses recientes quedan sesgados (el PSI de la duración de diciembre de 2025 dio 0,95 por este motivo y no por un cambio real). |

**Consecuencias de diseño:** las entradas y el puntaje se vigilan cada mes **con todos los contratos** (no necesitan etiquetas); el desempeño se mide cada trimestre **solo con cohortes completas**; y todo se compara con **el mismo mes o trimestre del año anterior**, nunca contra un promedio anual.

## 2. Qué se mide, cada cuánto y cuándo salta cada alarma

| Señal | Frecuencia | Se compara con | Aviso | Alarma |
|---|---|---|---|---|
| **PSI de 10 entradas** (precio base, duración, días de publicación a firma, días de firma a inicio, longitud del objeto, modalidad, tipo, sector, origen de recursos, UNSPSC) | Mensual | El mismo mes de 2022-2023 | 2 o más variables con PSI ≥ 0,10, o una con PSI ≥ 0,25: **revisar** | **Ninguna**: las entradas nunca reentrenan por sí solas |
| **PSI del puntaje** | Mensual (desde el segundo año) | El mismo mes del año anterior | ≥ 0,10 | ≥ 0,25 |
| **Tasa de positivos** (`y`, y `y_dias` como control del registro) | Trimestral, cohorte completa | El mismo trimestre del año anterior | Cambio de 5 puntos o más: investigar si es comportamiento o registro | Ninguna: solo orienta el diagnóstico |
| **ROC-AUC** | Trimestral, cohorte completa | El mismo trimestre del año anterior | Cae 0,03 o más y el IC 95 % por entidad excluye 0 | **Dos trimestres seguidos** |
| **Precisión del 1 % de mayor puntaje** | Trimestral, cohorte completa | La tasa base del trimestre | Menor que la tasa base (la lista de riesgo no supera a un contrato al azar) | **Dos trimestres seguidos** |
| **Concentración del top 1 %** | Trimestral | Una sola entidad con más del 50 % | Informativo: explica un cambio en el top, no es una alarma | — |

Una cohorte trimestral se evalúa con etiquetas solo si al menos el **95 %** de sus contratos ya maduró (`madura_en`, que reproduce exactamente `eval_madura`: 0 diferencias en 452.494 contratos). La madurez de un contrato es el fin planeado más 60 días, o la firma más 365 días si no tiene duración planeada.

### Por qué estos umbrales

- **PSI ≥ 0,10 / ≥ 0,25:** son los valores convencionales [SUPUESTO], y se contrastaron con los datos. Entre años consecutivos del entrenamiento (2021-2023) el PSI de las entradas ya tenía mediana 0,09 y el 29 % de los valores superaba 0,25, porque el SECOP II creció y cambió mucho. En 2024-2025 la mediana es 0,03 y solo el 1 % supera 0,25. Un PSI aislado, entonces, **avisa pero no decide**: por eso se pide más de una variable y nunca reentrena por sí solo.
- **Comparar contra el mismo trimestre del año anterior:** si se comparara contra un promedio anual, cada enero saltaría una alarma por la estacionalidad.
- **5 puntos en la tasa de positivos:** con las decenas de miles de contratos de un trimestre real el error estándar de la tasa es de menos de un punto, así que 5 puntos están muy por encima del ruido.
- **ROC-AUC con IC por entidad:** los contratos de una misma entidad están correlacionados, así que se remuestrean entidades y no contratos. Los intervalos resultan anchos (de ±0,04 a ±0,05 por trimestre). **Consecuencia honesta:** un monitor trimestral solo detecta caídas del ROC-AUC de unos 0,05 o más; una caída menor exigiría ventanas de un semestre [ESTIMADO a partir de esos intervalos].
- **Dos periodos seguidos para la alarma:** evita reentrenar por un único trimestre atípico.

## 3. Resultado de la simulación 2024-2025 [VERIFICADO]

Modelo congelado (entrenado con 2019-2023) recorrido mes a mes y trimestre a trimestre. El detalle está en `resultados/monitoreo_*.csv` y `resultados/monitoreo_resumen.md`.

| Qué mostró el monitor | Diagnóstico | Acción |
|---|---|---|
| **8 de 24 meses para revisar** por entradas (ene, mar, abr, jun y sep de 2024; ene, feb y jun de 2025). Las variables que más pesan son la duración planeada y el sector. | Cambios de composición frente a 2022-2023, mucho más suaves que la variación histórica entre años. | Observar. Sin impacto en el puntaje: su PSI máximo fue 0,094. |
| **Puntaje estable**: 12 de 12 meses normales de 2025 frente al mismo mes de 2024. | El modelo reparte los puntajes igual que el año anterior. | Ninguna. |
| **`y_dias` sube 6 a 9 puntos en los tres trimestres completos de 2025**, mientras `y` no. | Es consistente con un cambio en el registro de `dias_adicionados` (no verificado en v1), no con un cambio de comportamiento. Si ese fuera el caso y se usara `y_dias` como etiqueta, el monitor saltaría y reentrenar sería un error. | **No reentrenar**; mantener la etiqueta aprobada `y`. |
| **`y` baja 6,7 y 6,6 puntos en 2024-T1 y T2** frente a 2023. | [NO EVALUADO] la causa; el ROC-AUC de esos trimestres (0,771 y 0,737) no se degradó. | Seguir observando. |
| **El top 1 % de 2025-T3 es peor que el azar** (precisión 0,15 frente a una tasa base de 0,225), con el 82 % de los contratos de una sola entidad. | Cambio de comportamiento de esa entidad que el ROC-AUC global no muestra. | Aviso; si se repite en 2025-T4 pasa a alarma. Revisar el peso de la entidad en el modelo. |
| **Ninguna alarma de desempeño**: el ROC-AUC de 2025 frente al mismo trimestre de 2024 cambió −0,027, −0,002 y −0,009, con intervalos que incluyen 0. | El modelo se mantuvo estable. El −0,027 de 2025-T1 queda cerca del umbral de 0,03, pero su intervalo [−0,072, +0,016] incluye 0 y no genera aviso. | Sin reentrenamiento. |
| **2025-T4 incompleto** (91 % maduro): no se evalúa con etiquetas. | Censura: las etiquetas llegan con retraso. | Se evalúa cuando supere el 95 %. |

## 4. Cuándo se reentrena

1. **Alarma de desempeño:** el ROC-AUC cae dos trimestres seguidos, o el top 1 % queda peor que el azar dos trimestres seguidos.
2. **Cambio confirmado de la etiqueta o de su registro:** se corrige primero la etiqueta, como hizo la F1 con `y_dias`, y luego se reentrena.
3. **Revisión programada una vez al año**, cuando maduran los contratos del año anterior (hacia abril): se entrena un retador con una ventana que crece y se compara con el campeón. Se reentrena aunque no haya alarma, solo si el retador gana según la regla de la sección 5.
4. **Cambio estructural conocido** (una norma, una migración de plataforma) apoyado por un PSI sostenido de varias entradas.

**Nunca se reentrena solo por un PSI alto**, ni sin diagnosticar primero si el cambio es de comportamiento o de registro.

## 5. Cómo se valida y se pone en producción la versión nueva

1. **Datos:** reconstruir la base con el corte nuevo, pasar `calidad.py` y revisar la lista de fuga (`docs/LISTA_FUGA.md`) con lo aprendido.
2. **Retador:** entrenar con la ventana que crece (de 2019 al último año completo), con el mismo `Pipeline` y la semilla 42. Los hiperparámetros se vuelven a buscar con los folds temporales.
3. **Comparación justa:** evaluar campeón y retador en los **mismos** contratos, los trimestres completos más recientes que ninguno de los dos vio, con las mismas métricas (PR-AUC como principal, ROC-AUC, precisión en el top) y con IC 95 % remuestreando entidades (`diferencia_campeon_retador`).
4. **Regla de promoción:** el IC 95 % de la mejora en PR-AUC excluye 0 **y** el ROC-AUC no empeora claramente más de 0,01. Si el retador es peor, se descarta; si no hay evidencia, no se promueve. También se revisan las entidades no vistas y el top 1 %.
5. **Empaquetar:** `models/modelo_prorroga_v<N>.joblib` y su `.json` con las versiones del entorno, las features, el sha256, los percentiles y umbrales de riesgo, y la lista de entidades vistas. `cargar_modelo` rechaza un artefacto con otra versión de features, de scikit-learn o con un sha256 distinto. Versión nueva menor (reentrenamiento con las mismas features) o mayor (cambian las features) [EXTRA].
6. **Desplegar:** rama y PR revisado por otro integrante. La app usa el modelo de mayor versión que encuentre en `models/` y la plataforma se redespliega con el `push`. Se verifica la paridad entre API y base, la aditividad de SHAP y los tres ejemplos de la app.
7. **Volver atrás:** el campeón anterior se conserva un trimestre; revertir el commit devuelve el modelo anterior.

**Demostración [VERIFICADO]:** campeón (2019-2023) frente a retador (2019-2024), evaluados en 80.652 contratos de 2025-T1 a T3. PR-AUC 0,544 frente a 0,542, diferencia −0,002 con IC [−0,020, +0,032]; ROC-AUC 0,746 frente a 0,764, diferencia +0,018 con IC [+0,008, +0,028]. Decisión de la regla: **no promover, sin evidencia suficiente de mejora**. El PR-AUC, que decide, no mejora de forma clara; el ROC-AUC del retador sí es algo mayor, pero en la regla solo actúa como salvaguarda (que no empeore) y no como criterio de promoción.

## 6. Limitaciones

- **La simulación entrena con las etiquetas finales de 2019-2023** [SUPUESTO]. En producción real, el modelo solo habría visto las que estaban disponibles el día de la puesta en marcha.
- **Los umbrales se fijaron conociendo el comportamiento de 2024-2025**; no hay un periodo aparte para validar las propias reglas del monitor [NO EVALUADO].
- **Retador con los hiperparámetros del v1** [VERIFICADO]: en la demostración campeón-retador, el retador se entrena con los mismos hiperparámetros del v1; no se repite la búsqueda del paso 2 de la sección 5. Las cifras de este documento son las del modelo v1 y se regeneran con `python -m src.monitoreo`.
- **Poca potencia trimestral:** solo se detectan caídas grandes del ROC-AUC (sección 2).
- **El PSI convencional no distingue un cambio benigno de uno dañino**; sin etiquetas solo sirve para investigar.
- **Solo está diseñado, no automatizado:** el monitoreo mensual y trimestral se correría como una tarea programada que consulte la API de datos.gov.co, calcule estas tablas y avise. Queda fuera del alcance de este taller [EXTRA].
- **Etiquetas con retraso:** el desempeño de un contrato se conoce meses después, así que una degradación se detecta con ese retraso.

## 7. Cómo reproducirlo

```
python -m src.monitoreo        # desde final-project/, con la base v1 en data/processed/v1_A/ (unos 70 s)
pytest tests/test_monitoreo.py
```

Escribe `resultados/monitoreo_mensual.csv`, `monitoreo_trimestral.csv`, `monitoreo_psi_historico.csv`, `monitoreo_campeon_retador.json` y `monitoreo_resumen.md`. El borrador de la sección 3.9 del notebook está en `notebooks/_borradores/09_reajuste.ipynb`.
