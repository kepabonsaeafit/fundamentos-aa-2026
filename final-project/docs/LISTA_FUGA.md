# Lista de fuga de información (SECOP II)

> **Versión 1**, 23 de octubre de 2026: evidencia de los pasos 2 y 3 de la F1, aprobados por el usuario en sus checkpoints. Redactada el 25 de octubre de 2026, al cerrar la F1, con un dato del paso 4 (24 de octubre).
> Reemplaza la tabla preliminar de `docs/PLAN.md`, sección 4.8. `COLUMNAS_FUGA` de `src/features.py` es su espejo ejecutable; los cambios que faltan en el código están en la sección 8.
> Leyenda: **[VERIFICADO]** visto en datos o código · **[SUPUESTO]** · **[ESTIMADO]** · **[NO EVALUADO]**.
> Los reportes citados están en `final-project/.agents/tasks/f1/` (carpeta local, no versionada): `paso2_fecha_inicio.md`, `paso3_etiqueta.md` y `paso4_estimacion_f2.md`.

**Regla:** el momento de la predicción es la fecha de firma. Una columna tiene fuga si el valor que devuelve la API hoy puede contener hechos posteriores a la firma, como una ampliación del plazo, una adición de valor o la ejecución del contrato.

**Estados:**

| Estado | Qué significa | Qué se permite |
|---|---|---|
| Confirmada | Hay evidencia de que el valor incluye hechos posteriores a la firma | Nunca es feature. Solo EDA o ningún uso |
| Sospechosa | Hay indicios de que se actualiza después de la firma, sin prueba completa | Se trata como confirmada: no es feature ni respaldo |
| Etiqueta | Define `y` | Solo etiqueta |
| Por verificar | No se ha evaluado; se supone que no cambia después de la firma | Feature con ablación en J3, o excluida |
| Descartada (segura con limitación) | Se verificó y no hay diferencia por `y`, pero el método no ve todos los casos | Feature, con ablación obligatoria en J3 |

## 1. Cambios frente a la tabla preliminar (`PLAN.md`, 4.8)

| Columna | Antes | v1 | Evidencia |
|---|---|---|---|
| `fecha_de_inicio_del_contrato`, `dias_firma_a_inicio`, `fin_planeado_cruza_anio` | Por verificar | **Descartada (segura con limitación)** | Paso 2 (sección 6) |
| `duraci_n_del_contrato` | Por verificar | **Sospechosa** | Paso 3 (sección 4) |
| `fecha_de_fin_del_contrato` | Confirmada | Confirmada otra vez | Paso 3 (sección 2) |
| `dias_adicionados` | Etiqueta | Etiqueta, con sesgo de captura documentado | Pasos 3 y 4 (sección 3) |
| Modificaciones (`u8cx-r425`) | Etiqueta (el conjunto entero) | Etiqueta, columna por columna; solo cuenta `Publicado` | Paso 3 (sección 3) |
| `proposito_modificacion`, `sustituto_version` | Sin clasificar | Datos personales: no se descargan | Pasos 2 y 3 (sección 7) |
| `descripcion` (Modificaciones) | Sin clasificar | No aporta: no se descarga | Paso 3 (sección 7) |

## 2. Fuga confirmada

| Columna o derivado | Conjunto | Por qué es fuga | Evidencia | Uso permitido |
|---|---|---|---|---|
| `fecha_de_fin_del_contrato` y derivados (`dur_contrato_dias`, `razon_dur`) | Contratos | Incluye las ampliaciones del plazo | Exploración: la duración real sobre la planeada tiene mediana 1,44 con adición y 1,00 sin ella; ROC-AUC univariado 0,71. **F1, paso 3** [VERIFICADO]: es el fin de la última modificación publicada, o 1 día antes, en 1.520 de 1.521 contratos con modificaciones publicadas (565 iguales, 955 un día antes, 1 excepción de -5 días). Paso 2: va 1 día antes que el fin máximo de Modificaciones en 362 de 560 contratos | Solo EDA |
| `estado_contrato` | Contratos | Estado actual ("Modificado" es un estado final) | ROC-AUC univariado 0,94 (exploración) | Solo EDA |
| `valor_del_contrato` y derivados (`log_valor_contrato`, `razon_valor`) | Contratos | Incluye las adiciones de valor | Valor actual sobre el precio base: 1,18 con adición frente a 1,00 sin ella (exploración) | Solo EDA |
| 6 montos de fuentes de recursos (`presupuesto_general_de_la_nacion_pgn`, `sistema_general_de_participaciones`, `sistema_general_de_regal_as`, `recursos_propios_alcald_as_gobernaciones_y_resguardos_ind_genas_`, `recursos_de_credito`, `recursos_propios`) | Contratos | Su suma es el valor actual, que ya incluye las adiciones | La suma es igual a `valor_del_contrato` en el 80,6 % de los contratos; cociente sobre el precio base de 1,26 con adición en el valor frente a 1,00. Sumarlos a las 39 `FEATURES` sube el ROC-AUC de 0,751 a 0,807 (F0, base v0, `y_dias`; `PLAN.md`, sección 0, puntos 3 y 9). En la v1: ROC-AUC limpio 0,781; con 6 montos 0,917; con fuga evidente 0,984 | Solo EDA |
| Pagos, facturado, amortizado, pendientes y saldos (8 columnas) y `dias_firma_a_ult_actualizacion` | Contratos | Ejecución posterior a la firma | Pendientes: ROC-AUC univariado 0,63 (exploración) | Ninguno |
| `ultima_actualizacion` | Contratos | Fecha de la última edición del registro | ROC-AUC univariado 0,74 (exploración) | Ninguno |
| Liquidación y reversión (6 columnas) | Contratos | Posteriores al contrato | Por definición | Ninguno |
| `fecha_de_notificaci_n_de_prorrogaci_n` | Contratos | Registra la propia ampliación | Por definición | Ninguno |
| `visualizaciones_del`, `estado_del_procedimiento`, `id_estado_del_procedimiento`, `estado_resumen`, `fase`, `estado_de_apertura_del_proceso`, `fecha_de_ultima_publicaci` | Procesos | Estados actuales o conteos acumulados | La variante C de la exploración usó `visualizaciones_del` y sus cifras (0,809 / 0,436) no se citan (`PLAN.md`, sección 0, punto 1) | Ninguno |
| Ejecución (`mfmm-jqmq`) y Adiciones (`cb9c-h8sn`), conjuntos enteros | Otros | Registran hechos posteriores a la firma | Por definición | Ninguno |
| `valor_modificacion` | Modificaciones | Es el valor de la versión del contrato, igual a `valor_del_contrato` en el 75 % de las filas | Exploración | Ninguno |

**Contraste global** (exploración, test 2025; F0, base v0, `y_dias`): con las columnas posteriores a la firma, el mismo modelo pasa de 0,73 / 0,43 a **0,95 / 0,84** de ROC-AUC / PR-AUC. En la v1 (test 2024): ROC-AUC limpio 0,781; con 6 montos 0,917; con fuga evidente 0,984.

**Decisión:** confirmada desde la F0; la F1 solo agrega evidencia para `fecha_de_fin_del_contrato`.

**Qué hacer después:**
- **F2:** las columnas de Contratos con uso "solo EDA" se descargan a `data/raw/`; las demás no se descargan (`PLAN.md`, 4.2).
- **F3:** la base v1 no contiene ninguna columna de esta tabla. Lo comprueba el chequeo 1 de `src/calidad.py` (`FEATURES ∩ COLUMNAS_FUGA = ∅`).
- **EDA (F4):** muestra el contraste 0,95 / 0,84 (F0, base v0, `y_dias`) y la razón de duración como evidencia de fuga en §3.3 y §3.6. En la v1: ROC-AUC limpio 0,781; con 6 montos 0,917; con fuga evidente 0,984.
- **App (G1):** construye solo `FEATURES` e ignora estas columnas aunque la API las devuelva actualizadas.

## 3. Etiqueta (solo etiqueta)

**Definición aprobada** (checkpoint del paso 3): `y = 1` si `dias_adicionados > 0` **o** existe una modificación con estado `Publicado`, `dias_extendidos > 0` y `fecha_de_aprobacion > fecha_de_firma + 1 día`. El objetivo se llama **ampliación del plazo**, porque las suspensiones y los ajustes de fecha cuentan como positivos (cerca del 2,4 % de los contratos positivos).

| Columna | Conjunto | Hallazgo de la F1 | Uso permitido |
|---|---|---|---|
| `dias_adicionados` | Contratos | Copia los días de la última modificación publicada (1.499 de 1.521 contratos, 98,6 %) y vuelve a 0 cuando después se publica otra sin días, como una terminación o una liquidación. Eso explica 390 de los 392 contratos con extensión que `y_dias` no marca. `y_dias` captura el 43,4 % de los positivos de la etiqueta aprobada en 2019-2024 y el 80,0 % en 2025 (p = 2·10⁻⁷) [VERIFICADO, paso 3] | Solo etiqueta (`y_dias` y parte de `y`) |
| `estado_modificacion`, `dias_extendidos`, `fecha_de_aprobacion`, `fecha_fin_contrato` | Modificaciones | Solo `Publicado` tiene efecto: el fin vigente en Contratos es el de la última publicada en 1.520 de 1.521 contratos, y ninguna de las 19 no publicadas con un fin mayor se refleja en Contratos. Solo 3 de 1.546 extensiones se aprobaron, como mucho, 1 día después de la firma (todas en un contrato) [VERIFICADO, paso 3] | Solo etiqueta |
| `id_contrato`, `identificador_modificacion`, `version_anterior_del_contrato`, `fecha_version` | Modificaciones | Sirven para unir, ordenar y deduplicar (10,4 % de duplicados exactos en el paso 3 y 15,4 % en el paso 4) y para el corte `fecha_version <= corte` | Solo etiqueta |
| `y`, `y_dias`, `y_modif` | Base | Etiquetas | Solo etiqueta y sensibilidad |

**Sesgo de captura en la población** [VERIFICADO, paso 4]: entre el 14 y el 24 de octubre, los contratos 2019-2025 con `dias_adicionados > 0` bajaron de 602.328 a 600.674 (-1.654, -0,27 %) sin que cambiara el número de contratos. Es el mismo mecanismo de la vuelta a 0.

**Qué hacer después:**
- **F2:** Modificaciones se descarga con el filtro de servidor `estado_modificacion = 'Publicado' AND dias_extendidos > 0 AND fecha_version <= corte` y solo con las 8 columnas de esta tabla.
- **F3 (`src/etiqueta.py`):** deduplicar las filas exactas, calcular `y_modif` con la definición aprobada, `y = y_dias | y_modif`, y guardar `y_dias` e `y_modif` para la sensibilidad.
- **EDA y G3:** el salto de 2025 en `y_dias` es un efecto de registro, no una deriva real. Con la etiqueta aprobada, la tasa de 2019-2024 queda entre 19,5 % y 26,5 % sin diferencias entre años (χ², p = 0,56).
- Ninguna de estas columnas entra a `FEATURES`.

## 4. Fuga sospechosa

| Columna | Conjunto | Por qué es sospechosa | Evidencia | Uso permitido |
|---|---|---|---|---|
| `duraci_n_del_contrato` | Contratos | Se actualiza en parte después de la firma | Con unidad "Día(s)" (muestra de la exploración, 2019-2025), es igual al plazo actual (fin menos inicio, ±1 día) en el **24,0 %** de los `y_dias = 1` (2.736 de 11.385; IC 95 % de Wilson [23,3, 24,8], calculado para este documento). El plazo actual de un `y_dias = 1` ya incluye la extensión, así que sin actualización esa coincidencia no podría darse. En los `y_dias = 0` coincide en el 41,2 % (28.935 de 70.287) [VERIFICADO, paso 3] | Ninguno. Se descarga a `data/raw/` solo para documentarla |

- **Sin prueba estadística por grupo:** el argumento es por construcción. Comparar el 24,0 % con el 41,2 % no mide la fuga, porque en los `y_dias = 0` la duración y el plazo actual coinciden sin necesidad de actualización.
- **Decisión** (checkpoint del paso 3): pasa a sospechosa. Nunca fue feature (no está en `COLS_CONTRATO`) y sale del respaldo de la duración en la censura (`PLAN.md`, 4.5).
- **Qué hacer después:** F3 calcula `eval_fin_planeado` solo con la duración del proceso; si falta, `eval_madura = False`. La app no la usa. Se agrega a `COLUMNAS_FUGA` (sección 8).

## 5. Por verificar (la F1 no las evaluó)

| Columna o feature | Conjunto | Duda | Evidencia previa | Uso permitido |
|---|---|---|---|---|
| `el_contrato_puede_ser_prorrogado` | Contratos | ¿Se edita después de la firma? Es la más sospechosa por su nombre | ROC-AUC univariado 0,561; tasa de `y_dias` 0,188 con "Si" frente a 0,107 con "No"; parece un proxy de la modalidad (`PLAN.md`, sección 0, punto 10) | Solo ablación (excluida de `FEATURES`) |
| Conteos de competencia (las 7 `p_*` de H2) y `valor_total_adjudicacion` | Procesos | ¿Cambian después de la firma? | Quitar las 7 de competencia deja 0,750 / 0,334, así que no aportan (F0) [NO EVALUADO como fuga] | Feature, con ablación en J3 |
| `log_pago_adelantado` (`valor_de_pago_adelantado`) | Contratos | ¿Se actualiza después de la firma? | Marcada "por verificar" en `src/features.py` [NO EVALUADO] | Feature, con ablación en J3 |

**Qué hacer después:** J3 hace la ablación con y sin estas columnas, y el chequeo 7 de `src/calidad.py` alerta si alguna feature pasa de 0,85 de ROC-AUC univariado.

## 6. Fuga descartada: segura con limitación

| Columna o derivado | Conjunto | Uso permitido |
|---|---|---|
| `fecha_de_inicio_del_contrato` | Contratos | Entrada de `construir_features` |
| `dias_firma_a_inicio` | Derivada de Contratos | Feature, con ablación obligatoria en J3 |
| `fin_planeado_cruza_anio` | Derivada de Contratos y Procesos | Feature, con ablación obligatoria en J3 |

**Por qué se revisó:** de esta fecha dependen dos de las features que explican la ventaja del modelo sobre la regla por entidad (ROC-AUC +0,037 en la F0).

**Método** (paso 2): en 600 contratos de la base v0, se compara la fecha de inicio de la primera versión en Modificaciones con el valor actual en Contratos. La primera versión es la fila con menor `version_anterior_del_contrato` y, a igualdad, la de `fecha_version` más antigua. Ordenar por `fecha_creacion` o solo por `fecha_version` clasifica igual todos los contratos.

**Resultado** [VERIFICADO, paso 2; IC 95 % de Wilson]:

| Grupo | Comparables | Sin filas en Modificaciones | La fecha difiere | Cambio real |
|---|---|---|---|---|
| `y = 1` | 226 | 0 | 39 = **17,3 %** [12,9, 22,7] | 1 = 0,4 % [0,1, 2,5] |
| `y = 0` al azar | 134 | 40 de 174 (23 %) | 18 = **13,4 %** [8,7, 20,2] | 0 = 0 % [0, 2,8] |
| Discordantes (`y_dias = 0` con una extensión en Modificaciones) | 200 | 0 | 34 = 17,0 % [12,4, 22,8] | 4 = 2,0 % [0,8, 5,0] |

| Prueba | Diferencia | p (z) | p (Fisher) |
|---|---|---|---|
| La fecha difiere: `y = 1` frente a `y = 0` al azar (principal) | +3,8 pp | 0,34 | **0,37** |
| Desfase de 1 día: `y = 1` frente a `y = 0` al azar | +3,4 pp | 0,39 | 0,45 |
| Cambio real: con extensión (`y = 1` y discordantes) frente a `y = 0` al azar | +1,2 pp | 0,21 | 0,60 |
| Cambia `dias_firma_a_inicio`: `y = 1` frente a `y = 0` al azar | +4,6 pp | 0,25 | 0,29 |
| Cambia `fin_planeado_cruza_anio`: `y = 1` frente a `y = 0` al azar | -0,7 pp | 0,19 | 0,37 |

- **Casi todas las diferencias son un desfase de 1 día** (86 de 91). La fecha es la misma en todo el historial de Modificaciones y Contratos siempre muestra el día anterior. El fin del contrato tiene el mismo desfase en 362 de 560 contratos. La causa más probable es la zona horaria [SUPUESTO]. Para el modelo no importa, porque el entrenamiento y la app leen la fecha de Contratos.
- **Cambios reales** (la fecha cambia dentro del historial): 5 de 560 contratos (0,9 %), de +2 a +122 días, todos hacia después. Como mucho tocan el 2,7 % de los contratos con extensión (cota de Wilson). Aunque el modelo los aprovechara del todo, el ROC-AUC subiría como mucho ~0,007, frente a la ventaja de +0,037 sobre la regla [ESTIMADO].
- **Potencia** [ESTIMADO]: con 226 frente a 134 comparables, la prueba principal detecta con un 80 % de potencia diferencias de unos 11 pp o más.
- **Limitación** [NO EVALUADO]: el método no ve los cambios hechos antes de la primera modificación, que aparece en mediana 105 días después de la firma en los `y = 1` y 182 en los `y = 0` al azar. Tampoco ve los contratos sin modificaciones (23 % de los `y = 0` al azar). El paso 3 no cambió la conclusión, pero encontró ajustes de fecha "por acta de inicio" (1,7 % de las extensiones) que se parecen a los cambios reales.

**Decisión del usuario** (checkpoint del paso 2, 23 de octubre de 2026, 21:08): fuga descartada en lo que este método puede ver; las dos features quedan como **segura con limitación**. No se hizo la ablación en la F1.

**Qué hacer después:**
- **J3** ("Ablación explícita", `PLAN.md`): HGB final sin `dias_firma_a_inicio` ni `fin_planeado_cruza_anio`, contra la regla, global y dentro de entidad. Si la ventaja cae de forma importante, se documenta como limitación en 3.6.
- **Código:** en `src/features.py` las dos features siguen como "por verificar" (sección 8).

## 7. Datos personales y columnas que no se descargan (no son fuga)

| Columna | Conjunto | Motivo | Decisión |
|---|---|---|---|
| `autor_version` | Modificaciones | ID de usuario (PII ampliada del 2026-10-21) | No se descarga |
| `sustituto_version` | Modificaciones | Su rango en la metadata (2.000.016 a 40.059.164) coincide con el de `autor_version`, así que puede ser un ID de usuario. La F1 no confirmó qué es | No se descarga y pasa a la lista de datos personales junto a `autor_version` (checkpoint del paso 2) |
| `proposito_modificacion` | Modificaciones | Texto libre que a veces trae el nombre del contratista. La etiqueta no lo necesita | No se descarga en F2 y pasa a la lista de datos personales. Si el EDA usa las filas de la F1, no publica el texto |
| `descripcion` | Modificaciones | Es el objeto del contrato, no describe la modificación | No se descarga |

`src/descarga.py` tiene un `assert` que impide pedir estas cuatro columnas [VERIFICADO].

## 8. Espejo en código: cambios pendientes (F3)

`COLUMNAS_FUGA` (`src/features.py`, `FEATURES_VERSION = "1"`) ya cubre la fuga confirmada de la sección 2 y las etiquetas. Para que refleje esta v1 faltan cambios de código, que van en un PR aparte, no en el commit de documentación:

1. Agregar a `COLUMNAS_FUGA` la columna sospechosa `duraci_n_del_contrato` y las columnas de etiqueta de Modificaciones que no están: `fecha_de_aprobacion`, `fecha_fin_contrato`, `fecha_version`, `identificador_modificacion` y `version_anterior_del_contrato`.
2. En `_TABLA`, pasar `dias_firma_a_inicio` y `fin_planeado_cruza_anio` de "por verificar" a "segura con limitación". No cambia `FEATURES`, pero sí `FEATURES_POR_VERIFICAR`, que define el grupo de "29 seguras" de la sección 0, punto 9 del plan.
3. Agregar `sustituto_version` y `proposito_modificacion` a `PII_AMPLIADA` en `src/config.py`.

Cada hallazgo nuevo sube la versión de este archivo y se refleja en `COLUMNAS_FUGA` en el mismo PR.
