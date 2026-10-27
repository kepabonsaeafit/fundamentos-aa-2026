# Interpretaciones EDA para el modelo

Documento de trabajo para cerrar las lecturas pendientes de `notebooks/03_eda.ipynb` (§4.3, §5, §6.1–§6.7, §7.2 y §8). No modifica el notebook.

Cada **Decisión** describe lo que hace el pipeline v1, verificado contra `src/features.py`, `src/preprocesamiento.py`, `src/modelo.py`, `src/modelado.py` y `src/evaluacion.py`, con archivo y línea. Las recomendaciones del EDA que el pipeline no aplica quedan en **Alternativas** con la razón. Las cifras del EDA salen de `docs/eda_tablas.md` y se citan por tabla; para §6.1–§6.5, §6.7 y §7.2 la **Evidencia principal** son las tablas `X.b` sobre train 2019–2023, con la tabla original 2019–2025 como contraste; §6.6 usa test 2024 porque la tasa histórica se calcula con 2019–2023 y se evalúa fuera de muestra. Las tablas `X.b` usan el train completo (n = 261.831), no solo los 261.249 maduros con los que entrena el modelo; describen el tramo, no el ajuste, y las diferencias en pp son frente a la tasa base del subconjunto. Las cifras de la cadena de fuga de §8 salen de `resultados/`. Las hipótesis H1–H7 son las de `docs/01_definicion_problema.md:40-48`. El lenguaje es asociativo: “se asocia con”, “es consistente con”.

## §4.3 Madurez y etiqueta

**Decisión:** Usar `y` (ampliación del plazo; incluye suspensiones y ajustes de fecha, `docs/01_definicion_problema.md:26-29`) como etiqueta y filtrar `eval_madura = True` en entrenamiento y evaluación. El filtro está en `split_temporal` (`src/evaluacion.py:38-41`): ambos índices exigen `base["eval_madura"]`; el modelo se ajusta con esos índices y con `base["y"]` (`src/modelo.py:78-79`). Los no maduros no se usan como negativos ni se imputan, y `eval_madura` no entra como feature (no está en `FEATURES`, `src/features.py:85-87`). La etiqueta es `y = y_dias | y_modif` (`src/etiqueta.py:72`; metadatos en `src/modelo.py:51`).

**Alternativas:**
- Incluir todos los contratos y usar `eval_madura` como feature o peso.
- Imputar `y = 0` a los no maduros.
- Modelo de supervivencia que use la censura (fuera de alcance): requeriría modelar el tiempo hasta la ampliación y los contratos que aún no la han tenido; el alcance v1 es clasificación binaria con contratos maduros (`docs/01_definicion_problema.md:18-24`, `34`).

**Por qué:** Un contrato no maduro no ha tenido tiempo de que se observe una ampliación; tratarlo como negativo confunde censura con ausencia de evento. La pérdida de muestra es pequeña: en train 2019–2023 solo 582 de 261.831 contratos (0,22 %) no son maduros (suma de `n no maduros` de la Tabla 4.3.4). La etiqueta aprobada ya incorpora las ampliaciones publicadas en Modificaciones (`src/etiqueta.py:71-72`).

**Evidencia:** Tabla 4.3.2: 3.455 no maduros (0,8 % del total de 2019–2025), con `y` 22,5 % frente a 27,7 % en maduros. Tabla 4.3.4: en 2019–2024 (n=1.040 no maduros), 12,4 % en no maduros frente a 27,8 % en maduros; en 2025 (n=2.415), 26,8 % frente a 27,1 %. Tabla 4.3.5: los no maduros de 2019–2024 se reparten entre comodato (305; `y` 1,6 %), Otro (398; `y` 7,0 %) y el resto de tipos (337; `y` 28,5 %).

**Cautela:** La comparación de 2025 no controla por duración; no se afirma que la censura no influya. La exclusión se sostiene por el diseño —tiempo de observación insuficiente—, no por esa comparación.

## §5 Etiqueta `y` frente a `y_dias` y `y_modif`

**Decisión:** Usar `y` como objetivo (`src/etiqueta.py:72`; `src/modelo.py:51`, `79`). `y_dias` y `y_modif` no entran como features: no están en `FEATURES` (`src/features.py:85-87`) y se conservan solo para análisis de sensibilidad. La etiqueta v1 es la unión `y = y_dias | y_modif`.

**Alternativas:**
- Usar `y_dias` sola.
- Usar `y_modif` sola.
- Entrenar dos modelos separados o un modelo multi-etiqueta.

**Por qué:** `y_dias` capta solo el 48,2 % de los positivos (Tabla 5.4): deja fuera 64.759 contratos con `y=1` que solo aparecen en Modificaciones. `y_modif` coincide con `y` salvo 150 contratos (Tabla 5.3), así que la etiqueta descansa casi entera en Modificaciones; usar solo `y_dias` perdería más de la mitad de los positivos, y usar solo `y_modif` perdería los 150 que solo tienen días adicionados. La unión conserva los dos mecanismos.

**Evidencia:** Tabla 5.1: `y` 27,6 %, `y_dias` 13,3 % y `y_modif` 27,6 %; diferencia `y − y_dias` +14,3 pp. Tabla 5.2: 64.759 contratos con `y=1, y_dias=0` y 0 con `y=0, y_dias=1`. Tabla 5.3: 150 contratos con `y=1, y_modif=0`; ninguno con `y=0, y_modif=1`. Tabla 5.4: dentro de `y=1`, 48,2 % tiene `y_dias=1` (60.289 de 125.048).

## §6.1 H1. Los procesos apresurados se asocian con más ampliaciones del plazo

**Resultado: parcial** frente a H1 (`docs/01_definicion_problema.md:42`). La hipótesis espera que los procesos más apresurados (menos días entre publicación y firma) terminen más con el plazo ampliado. En train, el tramo de 0 días queda por debajo de la base (26,6 % frente a 28,8 %) y el máximo está en 31–60 días (40,1 %); la asociación existe, pero no es monótona ni en la dirección de “más apresurado, más ampliación”.

**Decisión:** `dias_publicacion_a_firma` entra como numérica (`src/features.py:75`); se calcula como `(firma - publicación).dt.days.clip(-30, 1000)` (`src/features.py:269`). En la familia `arbol` del pipeline v1 (`src/modelo.py:27`, `35-39`) las numéricas pasan crudas (`src/preprocesamiento.py:173`) y el HGB maneja los NaN de forma nativa (`src/preprocesamiento.py:8-10`). No hay imputación por mediana ni indicador `sin_publicacion` en v1; esos pasos solo existen en la familia `distancia` (`src/preprocesamiento.py:180-189`), que no es el artefacto.

**Alternativas:**
- Imputar la mediana y agregar un indicador de faltante (el EDA lo llama `sin_publicacion`; en `distancia` se implementa como `dias_publicacion_a_firma_faltante`, `src/preprocesamiento.py:119-136`): descartada en v1 porque el HGB maneja NaN de forma nativa (`src/preprocesamiento.py:8-10`) y el faltante es 0,08 % en 2019–2025 (380 de 452.494) y 11 contratos en train.
- Excluirla y dejar que el calendario capture el proceso.
- Usar solo tramos (0, 1–7, 8–15, …): el EDA lo recomendaba para modelos lineales o kNN por la relación no monótona; en v1 entra la numérica cruda.
- Familia `distancia` (kNN/SVM) con imputación, indicador y escalado: existe en `src/modelado.py:51-57` y `notebooks/05_modelado.ipynb`, pero no es el artefacto v1.

**Por qué:** La asociación no es lineal: se concentra en 31–120 días y cae en los extremos. El dato está disponible al firmar y el faltante es mínimo.

**Nota de texto:** el notebook dice 229 contratos sin fecha de publicación; la tabla ejecutada sobre 2019–2025 da 380, así que el texto debe pasar de **229 a 380** (en train 2019–2023 quedan 11).

**Evidencia principal (train 2019–2023):** Tabla 6.1.1.b (n = 261.831; y = 28,8 %): 0 días 26,6 % (n=91.896; -2,3 pp), 31–60 días 40,1 % (n=6.772; +11,3 pp), 61–120 35,7 % (n=2.449; +6,9 pp), >120 13,9 % (n=1.753; -15,0 pp); `Sin dato` 11.

**Contraste 2019–2025:** Tabla 6.1.1: 0 días 25,5 %, 31–60 40,0 %, >120 14,8 %; `Sin dato` 380 (20,0 %). La dirección se mantiene: el máximo sigue en 31–60 días y el tramo 0 sigue por debajo de la base.

## §6.2 H2. Poca competencia se asocia con más ampliaciones del plazo

**Resultado: contraria a la hipótesis** H2 (`docs/01_definicion_problema.md:43`). La hipótesis espera más ampliaciones cuando hay poca competencia; los datos muestran lo contrario en el número de respuestas: 0 respuestas queda en la base (28,9 %) y 10+ sube a 50,2 %, mientras que 1–3 respuestas queda por debajo (24,7 %).

**Decisión:** De las siete features de competencia marcadas `por verificar` (`src/features.py:77-83`), seis entran como numéricas crudas (`p_respuestas_al_procedimiento` y `p_proveedores_invitados` son las principales, `src/features.py:77`, `80`; cálculo en `src/features.py:270-273`); `p_proveedores_que_manifestaron` se descarta por ser constante 0 en el train (`src/preprocesamiento.py:19`, `43`). En la familia `arbol` no se imputan los ceros, no se aplica `log1p` y no se dicotomizan (`src/preprocesamiento.py:173`); el HGB maneja los NaN (`src/preprocesamiento.py:8-10`). J3 mantuvo las ocho `FEATURES_POR_VERIFICAR` en la lista (las siete de competencia y `log_pago_adelantado`, `src/features.py:66`); la ablación de quitarlas no cambió el PR-AUC 0,584 (`resultados/j3_decisiones.md:16`).

**Alternativas:**
- Excluirlas por la alta proporción de ceros.
- Usar solo una de las dos.
- Dicotomizar en “cero” frente a “alguna”, o en “10+” frente al resto.
- `log1p` + `RobustScaler` para kNN/SVM: existe en la familia `distancia` (`src/preprocesamiento.py:180-189`) y en los modelos de `src/modelado.py:51-57`, pero no en v1.

**Por qué:** La mayoría de los contratos no registra competencia, pero la cola se asocia con más ampliaciones; la relación no es lineal y los árboles pueden capturarla. La ablación de J3 muestra que el aporte no cambia el PR-AUC al quitarlas, así que se mantienen en `FEATURES` por ser información disponible al firmar y sin costo de imputación.

**Evidencia principal (train 2019–2023):** Tabla 6.2.1.b (n = 261.831; y = 28,8 %): 0 respuestas 28,9 % (n=235.700; +0,0 pp), 1 24,7 % (n=10.079; -4,1 pp), 2–3 24,7 % (n=7.220; -4,2 pp), 4–9 30,1 % (n=6.163; +1,2 pp), 10+ 50,2 % (n=2.663; +21,4 pp). Tabla 6.2.2.b: 0 invitados 28,7 % (n=255.976; -0,2 pp), 1–4 31,2 % (n=3.040; +2,3 pp), 5–9 39,9 % (n=722; +11,1 pp), 10+ 41,1 % (n=2.087; +12,3 pp).

**Contraste 2019–2025:** Tabla 6.2.1: 0 respuestas 27,5 %, 10+ 49,0 %; Tabla 6.2.2: 0 invitados 27,5 %, 10+ 41,7 %. La dirección se mantiene.

## §6.3 H3. La modalidad se asocia con la ampliación del plazo

**Resultado: confirmada** para H3 (`docs/01_definicion_problema.md:44`). La modalidad separa tasas muy distintas, desde 81,3 % en licitación pública de obra hasta 17,2 % en mínima cuantía; la hipótesis no fija una dirección, solo que la modalidad influye, y eso se observa.

**Decisión:** `modalidad_de_contratacion` y `justificacion_modalidad_de` entran como categóricas (`src/features.py:44-45`). En la familia `arbol` se codifican con `OrdinalEncoder(max_categories=200)`, que agrupa las categorías infrecuentes en un solo código (`src/preprocesamiento.py:167-168`, `172`); no hay one-hot ni una dicotomía directa/competitiva construida a mano. Las categorías nativas se descartaron porque la app explica con SHAP TreeExplainer (`resultados/j3_decisiones.md:10-13`).

**Alternativas:**
- Excluirla y dejar solo tipo de contrato o UNSPSC.
- Dicotomizar en contratación directa frente a competitiva.
- One-hot completo sin agrupar.
- Categorías nativas de HGB: +0,003 de PR-AUC cv, dentro del ruido, pero la app no puede explicarlas (`resultados/j3_decisiones.md:10-13`).

**Por qué:** La modalidad separa tasas muy distintas, pero puede estar confundida con el tipo de objeto y con el peso de la contratación directa. La codificación ordinal con tope de 200 deja que el HGB parta el espacio y evita estimaciones inestables de categorías con pocos casos.

**Evidencia principal (train 2019–2023):** Tabla 6.3.1.b (n = 261.831; y = 28,8 %): licitación pública de obra 81,3 % (n=561), concurso de méritos abierto 67,5 % (n=869), licitación pública 59,7 % (n=749), régimen especial 47,0 % (n=30.663), selección abreviada de menor cuantía 40,2 % (n=2.951), contratación directa 26,1 % (n=203.256), mínima cuantía 17,2 % (n=15.280).

**Contraste 2019–2025:** Tabla 6.3.1: licitación pública de obra 79,3 %, concurso de méritos 65,9 %, régimen especial 40,7 %, selección abreviada de menor cuantía 41,5 %, contratación directa 25,2 %, mínima cuantía 18,0 %. La dirección general se mantiene; **cambia el orden** entre régimen especial (47,0 %) y selección abreviada (40,2 %) en train, que en 2019–2025 aparecen como 40,7 % y 41,5 %.

## §6.4 H4. La anualidad presupuestal se asocia con el plazo

**Resultado: contraria a la hipótesis** H4 (`docs/01_definicion_problema.md:45`). La hipótesis espera más ampliaciones en el último trimestre; la tabla muestra lo contrario: 18,7 % en el último trimestre frente a 31,2 % en el resto en train (19,3 % frente a 29,7 % en 2019–2025).

**Decisión:** `mes_firma`, `firma_ultimo_trimestre`, `fin_planeado_cruza_anio` y `dias_firma_a_inicio` entran como numéricas (`src/features.py:68-72`). `mes_firma` se calcula como `firma.dt.month`, entero 1–12 (`src/features.py:253`), y en la familia `arbol` pasa crudo (`src/preprocesamiento.py:173`); no se trata como cíclica ni como one-hot. La versión cíclica (seno/coseno) existe solo en la familia `distancia` (`src/preprocesamiento.py:50`, `176`, `188`). `anio_firma` no está en `FEATURES` (`src/features.py:85-87`).

**Alternativas:**
- Excluir el calendario.
- Usar solo el último trimestre.
- Incluir el año de firma (descartado por deriva).
- `mes_firma` cíclica u one-hot, como recomendaba el EDA: descartada en v1; la versión cíclica solo se usa en `distancia` (`src/preprocesamiento.py:50`, `176`, `188`) y el HGB puede aislar cualquier mes con cortes sobre el entero.

**Por qué:** El patrón estacional es marcado y está disponible al firmar. La hipótesis sobre el último trimestre no se cumple en la dirección esperada, pero el mes sigue separando tasas y entra al modelo. No se incluye el año para no introducir deriva.

**Evidencia principal (train 2019–2023):** Tabla 6.4.2.b (n = 261.831; y = 28,8 %): último trimestre 18,7 % (n=50.186; -10,1 pp) y resto 31,2 % (n=211.645; +2,4 pp). Tabla 6.4.1.b: máximo en febrero 36,4 % (n=36.205), mínimo en octubre 16,1 % (n=16.814); enero 33,8 %, noviembre 16,2 %.

**Contraste 2019–2025:** Tabla 6.4.2: último trimestre 19,3 % frente a resto 29,7 %; Tabla 6.4.1: febrero 34,5 %, enero 33,9 %, noviembre 16,4 %. La dirección se mantiene y se acentúa.

## §6.5 H5. Los plazos subestimados se asocian con más ampliaciones

**Resultado: contraria a la hipótesis** H5 (`docs/01_definicion_problema.md:46`). La hipótesis espera que los plazos más subestimados se amplíen más; la tabla muestra lo contrario: (0, 0,5] de la duración típica tiene 20,5 % y (1,5, 2] tiene 41,1 % en train (19,9 % frente a 40,5 % en 2019–2025).

**Decisión:** `p_dur_dias` entra como numérica (`src/features.py:74`, `268`) y pasa por `DuracionLimpia` dentro del Pipeline (`src/preprocesamiento.py:17-18`, `55-76`, `172`): los valores 0 o negativos se vuelven NaN y generan el indicador `p_dur_dias_cero`, y el resto se recorta al cuantil 99,9 % aprendido del train (`src/preprocesamiento.py:40`, `62-67`); en el ajuste v1 (train maduro 2019–2023) ese tope es 1.259 días. `p_unidad_de_duracion` entra como categórica (`src/features.py:64`) y se codifica con el mismo `OrdinalEncoder`. No hay recorte fijo a 3.650 días: 3.650 es solo un corte descriptivo de la Tabla 4.3.3. La duración relativa a la mediana de la familia no está en `FEATURES`; solo existe para el EDA (`src/eda.py:748-760`).

**Alternativas:**
- Excluir la duración y usar solo el calendario.
- Usar tramos en vez de la numérica.
- Incluir solo la duración relativa: queda como alternativa opcional, fuera del pipeline v1; no está en `FEATURES` y solo se calcula en `src/eda.py:748-760`.
- Recorte fijo a 3.650 días, como recomendaba el EDA: descartado; el Pipeline usa el p99,9 aprendido del train (`src/preprocesamiento.py:17-18`, `40`), que se adapta a la distribución y no fija un corte arbitrario.
- `log1p` + `RobustScaler` para kNN/SVM: solo en la familia `distancia` (`src/preprocesamiento.py:180-189`), no en v1.

**Por qué:** La relación no es monótona: sube hasta 366–730 días y cae en >730; la relativa muestra un máximo en (1,5, 2]. La duración 0 no es un plazo real y se trata como faltante, no como cero, con su indicador.

**Evidencia principal (train 2019–2023):** Tabla 6.5.1.b (n = 261.831): (0, 0,5] 20,5 % (n=51.465; -8,4 pp), (0,5, 1] 28,4 %, (1, 1,5] 31,6 %, (1,5, 2] 41,1 % (n=33.771; +12,2 pp), (2, 3] 27,8 %, >3 27,7 %, <=0 25,2 %. Tabla 6.5.2.b: <=0 25,3 % (n=13.712), 1–30 22,8 %, 31–90 23,1 %, 91–180 28,9 %, 181–365 34,7 %, 366–730 59,6 % (n=2.222), >730 28,5 % (n=1.381), sin dato 11,3 %.

**Contraste 2019–2025:** Tabla 6.5.1: (0, 0,5] 19,9 %, (1,5, 2] 40,5 %; Tabla 6.5.2: 366–730 57,1 %, >730 22,1 %, <=0 24,7 %. La dirección se mantiene.

## §6.6 H6. Las entidades tienen hábitos

**Resultado: confirmada** para H6 (`docs/01_definicion_problema.md:47`). La tasa histórica de la entidad separa la tasa futura: T3 41,6 %, T2 20,5 %, T1 14,3 %, tasa 0 11,4 %, sin historia 11,5 % en test 2024.

**Decisión:** `nit_entidad` entra como categórica (`src/features.py:57`) y se codifica con `TargetEncoder(target_type="binary", smooth="auto", cv=StratifiedKFold(5, shuffle=True, random_state=config.SEED))` dentro del Pipeline (`src/preprocesamiento.py:151-155`, `190`); las entidades no vistas reciben la media global del train (`src/preprocesamiento.py:152-153`). Los proxies de entidad (`sector`, `orden`, `rama`, `entidad_centralizada`, `departamento`, `ciudad`) entran como categóricas (`src/features.py:58-63`) y, como el resto de categóricas, usan `OrdinalEncoder(max_categories=200)` (`src/preprocesamiento.py:167-168`). No se usa one-hot de entidad ni el nombre.

**Alternativas:**
- One-hot de entidad: inviable por cardinalidad (2.086 entidades en test 2024, Tabla 6.6.1).
- Excluir la entidad.
- Usar la tasa histórica calculada fuera del Pipeline: filtraría información y no se reajustaría con cada fold.
- Codificar `nit_entidad` con `OrdinalEncoder` como el resto: convertiría un código arbitrario en una escala ordinal; el `TargetEncoder` captura la tasa histórica, que es la señal de H6.

**Por qué:** La tasa histórica de la entidad se asocia con la tasa futura, y las entidades sin historia se separan con una tasa baja; el codificador debe aprenderla solo con train para no filtrar 2024. La cardinalidad alta hace inviable el one-hot. El `cv` interno evita fuga dentro del train.

**Evidencia:** Tabla 6.6.1 (test 2024; historia 2019–2023; n=88.546): T3 41,6 %, T2 20,5 %, T1 14,3 %, tasa 0 11,4 %, sin historia 11,5 %. Tabla 6.6.2: 439 entidades en cada tercil y 1.017 en tasa 0. Esta sección usa test 2024, no las tablas `.b` de train.

**Cautela:** La evidencia sale de test 2024, no de train; la tasa histórica se calcula solo con 2019–2023. No se usa 2024 para ajustar el codificador.

## §6.7 H7. El tipo de objeto importa

**Resultado: confirmada** para H7 (`docs/01_definicion_problema.md:48`). Los segmentos y familias UNSPSC, el tipo de contrato y la longitud del objeto separan tasas: segmento 85 40,3 % frente a 82 17,7 %; interventoría 71,9 % frente a comodato 18,6 %; `len_objeto` <=50 58,5 %.

**Decisión:** `tipo_de_contrato`, `unspsc_segmento`, `unspsc_familia` y `len_objeto` entran en `FEATURES` (`src/features.py:43`, `46-47`, `67`); las dos UNSPSC se derivan del código de categoría (`src/features.py:247-250`). Las tres categóricas se codifican con `OrdinalEncoder(max_categories=200)` (`src/preprocesamiento.py:167-168`, `172`); `len_objeto` es numérica cruda (`src/features.py:263`) y no se usa el texto crudo del objeto. `len_objeto` se conserva con seguimiento en la evaluación con entidades no vistas: es un indicador indirecto de la entidad (cargos de hospitales), no una fuga (`docs/EDA_INSIGHTS_PREPROCESAMIENTO.md:26-29`).

**Alternativas:**
- Excluir `len_objeto` y dejar solo tipo/UNSPSC.
- Usar el texto crudo del objeto.
- Agrupar todos los tipos raros en “Otro” a mano (el `OrdinalEncoder` ya agrupa las categorías infrecuentes en un código).
- Categorías nativas de HGB: descartadas por la explicación SHAP (`resultados/j3_decisiones.md:10-13`).

**Por qué:** Tipo y UNSPSC separan tasas muy distintas; `len_objeto` corto se asocia con objetos de personal de hospitales y puede estar capturando la entidad. Es señal disponible al firmar, pero su peso debe revisarse fuera de las entidades vistas.

**Evidencia principal (train 2019–2023):** Tabla 6.7.1.b: segmento 85 40,3 % (n=13.544), 72 38,6 % (n=4.408), 93 34,3 %, 80 28,5 %, 82 17,7 %. Tabla 6.7.2.b: familia 8510 45,8 % (n=9.672), 8110 41,0 %, 9315 38,5 %, 9314 32,7 %, 8011 28,8 %, 8111 21,6 %. Tabla 6.7.3.b: interventoría 71,9 % (n=723), obra 53,2 % (n=2.409), consultoría 45,3 %, prestación de servicios 28,2 % (n=224.529), compraventa 20,5 %, suministros 20,3 %, comodato 18,6 %. Tabla 6.7.4.b: <=50 caracteres 58,5 % (n=4.385), 51–100 35,5 %, 101–200 25,7 %, 201–400 29,6 %, >400 26,8 %.

**Contraste 2019–2025:** Tabla 6.7.1: 72 38,9 %, 85 38,5 %, 80 27,0 %, Otras 22,2 %; **cambia el orden** entre 85 y 72. Tabla 6.7.2: 8510 43,7 %, 8110 38,9 %, 9314 32,9 %, 9315 32,1 %; **cambian de orden 9315 y 9314 y entra 8111 en lugar de 8512**. Tabla 6.7.3: interventoría 71,8 %, obra 51,9 %, prestación de servicios 27,1 %, comodato 16,4 %; **Decreto 092 pasa de -2,3 pp a +6,6 pp** y sube en el orden. Tabla 6.7.4: <=50 58,0 % frente a 201–400 28,2 %; la relación sigue sin ser monótona. La dirección general se mantiene en las cuatro.

## §7.2 Tasa de y por tipo de documento del proveedor

**Decisión:** Conservar solo `tipodocproveedor` como feature categórica (`src/features.py:56`), codificada con `OrdinalEncoder(max_categories=200)` en la familia `arbol` (`src/preprocesamiento.py:167-168`, `172`); excluir la identidad y los datos del proveedor (documento, nombre, código, NIT adjudicado).

**Alternativas:**
- Incluir el historial de prórrogas del proveedor.
- Usar un identificador cifrado o hash.
- Usar el proveedor solo para personas jurídicas (NIT).

**Por qué:** La cédula de ciudadanía domina la muestra y es un dato personal; un puntaje por proveedor se convertiría en un ranking de personas. El tipo de documento no identifica al proveedor y las diferencias entre cédula y NIT son pequeñas (28,8 % vs 28,7 % en train).

**Evidencia principal (train 2019–2023):** Tabla 7.2.1.b (n = 261.831; y = 28,8 %): cédula de ciudadanía `y` 28,8 % (n=219.724), NIT `y` 28,7 % (n=40.593), Otro `y` 29,6 % (n=564), Otras `y` 43,7 % (n=950).

**Contraste 2019–2025:** Tabla 7.2.1: cédula de ciudadanía 84,3 % (n=381.242) con `y` 27,3 %; NIT 15,2 % (n=68.844) con 28,9 %; No Definido 0,2 % (n=730) con 60,5 %. **Cambia la comparación**: en train NIT y cédula quedan casi iguales (28,7 % vs 28,8 %), mientras que en 2019–2025 el NIT está 1,6 pp por encima; `Otras` mezcla No Definido y Cédula de Extranjería, así que su 43,7 % no es comparable. Tabla 7.2.2: las tasas anuales de las categorías pequeñas son inestables; “Otras” pasa de 0,0 % (n=2) en 2019 a 26,9 % (n=52) en 2025.

## §8 Fuga descriptiva

**Decisión:** Excluir de `FEATURES` las columnas de fuga confirmada y los agregados de Modificaciones (`src/features.py:141-178`); `estado_contrato` se usa solo para EDA, no como feature ni como respaldo. `el_contrato_puede_ser_prorrogado` sigue fuera de `FEATURES` por la decisión del PLAN (`src/features.py:55`); la ablación de J3 la respalda (agregarla mueve el PR-AUC 2024 en −0,0010, dentro del ruido; `resultados/j3_decisiones.md:23`).

**Alternativas:**
- Incluir `estado_contrato` con regularización.
- Usar las columnas solo para evaluación o para definir la etiqueta.
- Incluir los agregados de Modificaciones como features.

**Por qué:** Esas columnas contienen hechos posteriores a la firma; su AUC alto es consistente con fuga, no con señal utilizable al firmar. El AUC baja de 0,943 con `y_dias` a 0,820 con `y` porque cambió la etiqueta, no porque deje de ser fuga: “Modificado” tiene 76,9 % de positivos frente a 0,3 % de “En ejecución”, es decir, el estado se actualiza con el desenlace.

**Evidencia:** Tabla 8.1 (2019–2025): “Modificado” 76,9 % de `y` (n=81.391) frente a “En ejecución” 0,3 % (n=118.069) y “Aprobado” 0,2 % (n=10.997). Tabla 8.2: `estado_contrato` AUC 0,820 vs `y` y 0,943 vs `y_dias` (legacy 0,94); `razon_dur` 0,952; `mod_n` 0,9996; `dias_adicionados` 1,0000 vs `y_dias`.

**Cadena de fuga del modelo:** con el HGB limpio, ROC-AUC 0,781 → 0,917 → 0,984 y PR-AUC 0,584 → 0,851 → 0,967: el primer salto agrega los 6 montos del crudo de Contratos y el segundo la fuga evidente (estado, valor y fecha de fin). Cifras en `resultados/j3_extra_fuga_montos.csv` (filas `final_limpio`, `final_con_6_montos`, `final_con_fuga_evidente`) y en `resultados/j3_decisiones.md:24`.

**Cautela:** Los AUC de la Tabla 8.2 son descriptivos, no una métrica de validación; no hay train/test ni modelo en esa sección. La cadena de fuga es un contraste sobre la base v1_A con el crudo de Contratos unido por `id_contrato` solo para medir; esas columnas no entran en la base ni en `FEATURES`.
