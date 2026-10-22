# Plan del proyecto final: riesgo de prórroga en contratos públicos (SECOP II)

> Versión 3.2, con el calendario, los recortes y los límites de tiempo de Kevin, y con lo medido en la F0: entorno, hallazgo de SHAP, fuga de los montos y señal limpia (sección 0, puntos 7 a 11, y lo que de ellos se deriva en F0, J1-J3, G1 y riesgos).
> Leyenda: **[VERIFICADO]** visto en datos/código · **[SUPUESTO]** · **[ESTIMADO]** · **[NO EVALUADO]**.
> Cada tarea va marcada **[MÍNIMO]** (necesaria para cumplir el enunciado) o **[EXTRA]** (fuera del alcance salvo que sobre tiempo).

## Contexto

La exploración (`00_exploracion/`) eligió SECOP II: predecir, al firmar un contrato, la probabilidad de que termine prorrogado, para que veedurías y oficinas de control interno prioricen qué vigilar. El profesor aprobó el tema y pidió **justificar bien cómo los datos del contrato predicen la prórroga y qué ingresa el usuario**. Hay señal, pero más modesta de lo que decía la exploración (sección 0, punto 9): con las features limpias el HGB da 0,751 ROC-AUC / 0,333 PR-AUC en 2024 frente a 0,714 / 0,277 de la regla por entidad. Además, la etiqueta subestima los positivos, hay censura y la tasa salta de 12 % a 20 % en 2025.

**Fechas:**
- Quedan **10 días**.
- Meta interna: entregar el **sábado 31 de octubre**.
- Entrega oficial: **domingo 1 de noviembre**.
- Ensayo: **lunes 2**.
- Sustentación: **martes 3**.

**Equipo:**
- Kevin: datos + EDA; revisa los PRs y el notebook integrado.
- Jose: evaluación → preprocesamiento → modelado → integración del notebook.
- Geronimo: app → despliegue → reajuste.

El paralelismo se apoya en cuatro **contratos** (sección 2) y en una **base v0** compartida la noche del jueves 22.

## 0. Hallazgos de la lectura que cambian cosas

1. **La variante C de la exploración usó `p_visualizaciones_del`**, que está en la lista de fuga. Sus cifras (0,809 / 0,436 en 2024) pueden estar infladas; se recalculan sin esa columna antes de citarlas. [VERIFICADO en `exploracion_secop.ipynb`, `NUM_P`]
2. **El bootstrap de la exploración remuestrea contratos**, que están correlacionados dentro de cada entidad, así que los intervalos son optimistas. Aquí se remuestrea **por entidad**.
3. **Los 6 montos de fuentes de recursos son fuga confirmada.** [VERIFICADO con la muestra de la exploración, firma 2019-2025] Su suma es igual a `valor_del_contrato` en el 80,6 % de los contratos y, en los contratos con un evento de adición en el valor, el cociente sobre el precio base es 1,26 (1,00 sin adición), igual que el de `valor_del_contrato`. Quedan fuera de `FEATURES` y dentro de `COLUMNAS_FUGA`.
4. **La población 2019-2025 es de 4.501.433 contratos** (602.328 con días adicionados, 13,4 %) [VERIFICADO en `meta/poblacion_contratos_por_anio.csv`]. Se trabaja con una muestra del 10 % (sección 4.1).
5. **Jose y Geronimo no tienen los datos de la exploración** (`data/` no se versiona). La base v0 se comparte por Drive/OneDrive con sha256.
6. La rama base es **`master`**. `REGLAS_DEL_PROYECTO.md` dice `main` en las líneas 35 y 37; se corrige en el primer commit.
7. **SHAP y las categorías nativas del HGB.** [VERIFICADO con datos sintéticos, `scikit-learn` 1.9.0 + `shap` 0.54.1] `TreeExplainer` sobre un `HistGradientBoostingClassifier` con códigos ordinales y sin categorías nativas es exacto (error de aditividad 0,0000). Con `categorical_features=mask` no lanza error, pero las atribuciones son incorrectas (error de aditividad máximo 13,6 log-odds, medio 5,4). La exploración entrenó el HGB con categorías nativas (`exploracion_secop.ipynb`: `categorical_features=mask` sobre un `OrdinalEncoder` limitado a las 200 categorías más frecuentes), así que las cifras de referencia (0,763 / 0,334) dependen de eso. En J2-J3 el equipo decide entre (a) mantener las nativas y explicar con `shap.Explainer` genérico sobre `predict_proba` (funcionó en la prueba; más lento) o (b) quitar las nativas y usar `TreeExplainer`, midiendo cuánto baja el PR-AUC. La decisión se documenta en el notebook con Decisión / Alternativas / Por qué / Evidencia.
8. **Entorno.** Python 3.12 con venv en `final-project/.venv` y versiones fijas en `requirements.txt` (`scikit-learn` 1.9.0, `shap` 0.54.1, `streamlit` 1.66.1, `pytest` 9.1.1). El `requirements.txt` de la raíz dice Python 3.14.3; el del proyecto final usa 3.12 porque `shap` 0.54.1 exige Python ≥ 3.12 y es la versión instalada en la laptop de Kevin. Se omiten `jupyter` y `notebook`: traen `jupyterlab`, que no se instala en Windows sin rutas largas cuando la carpeta del repo es profunda (una ruta de 270 caracteres frente al límite de 260 en la carpeta de Kevin). `nbconvert` + `ipykernel` bastan para `jupyter nbconvert --execute` y para abrir notebooks en VS Code. **Pendiente:** Jose y Geronimo confirman que pueden usar Python 3.12 y Geronimo confirma que Streamlit Community Cloud lo ofrece [SUPUESTO].
9. **Señal limpia medida en la F0.** [VERIFICADO; muestra de la exploración, train 2019-2023 y test 2024 con 23.999 contratos y tasa base 0,123; montaje de la exploración, HGB sin ajustar] Regla por entidad: 0,714 ROC-AUC / 0,277 PR-AUC. HGB sobre las 39 `FEATURES`: 0,751 / 0,333 (2,72 veces la tasa base). HGB A original, con los 6 montos: 0,763 / 0,334. El HGB limpio le gana a la regla por +0,037 de ROC-AUC (IC 95 % con bootstrap por entidad, 200 repeticiones: [+0,011, +0,069]) y +0,056 de PR-AUC ([+0,003, +0,113]).
   - Los 6 montos inflan el resultado: quitarlos de la variante A baja el ROC-AUC de 0,763 a 0,742; sumarlos a las 39 `FEATURES` lo sube de 0,751 a 0,807 y el PR-AUC de 0,333 a 0,429, porque junto con el precio base delatan las adiciones de valor. Ese 0,807 / 0,429 se parece mucho a la variante C de la exploración (0,809 / 0,436), cuyas cifras probablemente están infladas por lo mismo y no se deben citar [SUPUESTO: C no se recalculó].
   - "Limpio" significa sin la fuga confirmada. Con solo las 29 `FEATURES` seguras (sin las 10 "por verificar") la ventaja sobre la regla deja de ser clara: ROC-AUC +0,027 [-0,002, +0,060] y PR-AUC +0,038 [-0,013, +0,099]. Las 7 de competencia (H2) no aportan: quitarlas deja 0,750 / 0,334, así que lo que aportan las 10 viene de las otras 3 [SUPUESTO, por descarte].
   - Los intervalos por entidad son mucho más anchos que los de la exploración, que remuestreaba contratos.
10. **`el_contrato_puede_ser_prorrogado` queda excluida de `FEATURES`** (por verificar; es la más sospechosa por su nombre). [VERIFICADO] Su señal es débil: tasa de `y` de 0,188 con "Si" (34.351 contratos) frente a 0,107 con "No" (133.646), ROC-AUC univariado 0,561, con el mismo signo cada año de 2019 a 2025. Dentro de "Contratación régimen especial" la diferencia es grande (0,379 frente a 0,121) y en "Contratación directa" casi nula (0,118 frente a 0,107), así que se parece a un proxy de la modalidad. Solo el 31 % de los positivos tiene "Si", y su proporción baja de 31,8 % (2019) a 11,7 % (2025). Sigue en la base y se obtiene con `construir_features(df, con_excluidas=True)`; J3 la evalúa en la ablación.
11. **Entidades no vistas y top-k, medidos en la F0.** [VERIFICADO; mismo montaje del punto 9: base v0, test 2024 con 23.999 contratos, HGB limpio sin ajustar, intervalos con bootstrap por entidad de 200 repeticiones]
   - **Entidades no vistas en train 2019-2023:** 235 de las 1.638 entidades del test, con 804 contratos (3,4 %) y 55 positivos; tasa base 0,068. La regla no tiene información (ROC-AUC 0,500 por construcción, PR-AUC 0,068). El HGB limpio da ROC-AUC 0,708 [0,584, 0,800] y PR-AUC 0,180 [0,083, 0,309] (2,63 veces la tasa base). Diferencia con la regla: ROC-AUC +0,208 [+0,084, +0,300] y PR-AUC +0,112 [+0,033, +0,220]. Con solo las 29 features seguras da 0,709 / 0,181, así que este resultado no depende de las 10 "por verificar". En entidades vistas la ventaja es la general: ROC-AUC +0,033 [+0,006, +0,063] y PR-AUC +0,055 [+0,001, +0,114]. Es la parte más sólida de la justificación del modelo, con la reserva de que son solo 55 positivos.
   - **Precisión en el top-k** (k = 240, 1.200 y 2.400 contratos; tasa base 0,123):

     | | Top 1 % | Top 5 % | Top 10 % |
     |---|---|---|---|
     | HGB limpio, precisión | 0,654 [0,511, 0,782] | 0,441 | 0,364 |
     | Regla (m=0), precisión | 0,438 [0,166, 0,534] | 0,432 | 0,310 |
     | Diferencia | +0,217 [+0,064, +0,466] | +0,008 [-0,085, +0,161] | +0,054 [-0,048, +0,127] |
     | HGB limpio, recall | 0,053 | 0,180 | 0,297 |
     | Regla (m=0), recall | 0,036 | 0,176 | 0,253 |

   - **La regla con m=0 es un rival débil en el top-k:** m se eligió por ROC-AUC y el top 1 % queda dominado por entidades con 1 de 1 positivos. Con la regla suavizada (sensibilidad medida en test, no para elegir m) la precisión sube: m=5 da 0,537 / 0,462 / 0,330 y m=20 da 0,537 / 0,468 / 0,343. Con m ≥ 5 el HGB solo gana claramente en el top 1 % (0,654 frente a 0,537); en el 5 % pierde frente a la regla (0,441 frente a 0,46-0,47) y en el 10 % queda casi igual. La ventaja del modelo en el top-k está solo en el 1 % más alto (240 contratos, intervalo ancho). `elegir_m` de J1 debe considerar también la precisión en el top-k.
   - **Códigos ordinales frente a categorías nativas** (contratos maduros de 2024): HGB nativo con las 200 categorías más frecuentes 0,752 / 0,336; HGB con códigos ordinales (el dummy, sin early stopping) 0,766 / 0,344. Los códigos no pierden en esta muestra (una sola corrida, sin intervalo); alimenta la decisión (a)/(b) del punto 7.
   - **Madurez:** filtrar por `eval_madura` casi no cambia las referencias (regla 0,714 → 0,715; HGB 0,751 → 0,752), porque la duración planeada mediana es de 120 días y el 97-99 % de los contratos de 2019-2025 ya venció.

---

## 1. Mapa de requisitos

| Req. | Qué pide | Dónde se cumple | Responsable |
|---|---|---|---|
| 3.1 Problema | Contexto, usuarios, decisión, datos | `docs/01_definicion_problema.md` → notebook §3.1, con la tabla "qué ingresa el usuario" (pedido del profesor) | Kevin |
| 3.2 Preprocesamiento | Limpieza, faltantes, atípicos, codificación, escalado, justificados | Limpieza determinista en `src/features.py` (Kevin) y `Pipeline` en `src/preprocesamiento.py` (Jose) → notebook §3.2 | Jose (+Kevin) |
| 3.3 EDA | EDA profundo + insights → decisiones | Notebook §3.3: H1-H7 con evidencia (**cómo los datos de la firma predicen la prórroga**), etiqueta, censura, deriva, fuga, decisión de privacidad del proveedor | Kevin |
| 3.4 Paradigma | ≥ 1 paradigma | Notebook §3.4: supervisado (clasificación). [EXTRA] complemento no supervisado | Jose |
| 3.5 Metodología y selección | Alternativas, comparación, hiperparámetros | Notebook §3.5 con folds temporales: kNN, árbol, RF, HGB, SVM | Jose |
| 3.6 Generalización | Datos no vistos, sin sobreajuste ni fuga | Notebook §3.6: test 2024 usado una vez, deriva 2025, entidades no vistas, `docs/LISTA_FUGA.md`, contraste "con fuga", SHAP global enlazado a H1-H7 | Jose (+Kevin en fuga) |
| 3.7 Métricas | Justificadas en términos de uso | Notebook §3.7: PR-AUC, precisión/recall en top-k, tasa base, regla por entidad | Jose |
| 3.8 Despliegue | Interfaz para no técnicos | `app/` + URL pública + notebook §3.8 (arquitectura, flujo, capturas) | Geronimo |
| 3.9 Reajuste | Detectar degradación, criterio, frecuencia, validar y promover | `src/monitoreo.py` + notebook §3.9 | Geronimo |
| Notebook | `model_development.ipynb` ejecutado | Raíz de `final-project/` | **Jose integra; Kevin revisa** |
| Repo y documentación | README, docs | `README.md`, `docs/` | Kevin (README) + todos |
| Video de 2 min | App desplegada | Enlace en el README | Geronimo (guion con todos) |
| Sustentación (50 %) | Defender todo | Ensayo del lunes 2 | Todos (cada uno domina todo) |

**"Qué ingresa el usuario"** (para §3.1 y §3.8): el usuario solo pega el número del contrato. La app consulta Contratos y Procesos en datos.gov.co, construye **solo** las variables disponibles al firmar (las mismas del entrenamiento) e ignora las columnas de la lista de fuga, aunque la API las devuelva actualizadas. La tabla del notebook tiene cuatro columnas: lo que pega el usuario → lo que consulta la app → las variables que construye → lo que se descarta y por qué.

---

## 2. Contratos entre las partes

Principio: **lo determinista y fila a fila va en `src/features.py`; todo lo que se aprende de los datos (imputación, codificación, escalado, target encoding) va dentro del `Pipeline` que se guarda en el artefacto.** Así la app reproduce exactamente el entrenamiento y el preprocesamiento no tiene fuga.

### 2a. Esquema de la base procesada

Archivo: `data/processed/base_contratos.parquet` + `base_contratos.meta.json` (fecha de corte, versión, definición de la etiqueta, dígito de muestreo, filas, sha256). Una fila por contrato y `id_contrato` único.

| Grupo | Columnas | Tipo | Uso |
|---|---|---|---|
| Llaves | `id_contrato`, `proceso_de_compra` | string | Únicas / join |
| Momento y split | `fecha_de_firma`, `anio_firma` | datetime64, int16 | Split temporal |
| Entidad | `nit_entidad`, `nombre_entidad` | string | Regla por entidad y bootstrap; el nombre solo se muestra |
| Entradas de Contratos | `COLS_CONTRATO` (sección 4.2) | tipadas | Solo las lee `construir_features` |
| Entradas de Procesos | `COLS_PROCESO` con prefijo `p_` | tipadas | Solo las lee `construir_features` |
| Etiquetas (prefijo `y`) | `y` (final), `y_dias` (`dias_adicionados > 0`), `y_modif` (Int8 con NA) | int8 | Objetivo y sensibilidad |
| Evaluación (prefijo `eval_`) | `eval_dur_planeada_dias`, `eval_fin_planeado`, `eval_madura`, `eval_en_procesos` | float32, datetime64, bool, bool | Censura y métricas por grupo; **nunca son features** |

La base **no contiene columnas de fuga**; las que se necesitan para demostrarla en el EDA se leen de `data/raw/`.
- **Base v0** (jueves 22): mismo esquema con la muestra de la exploración (167.997 contratos de 2019-2025, unos 24.000 por año de firma), `y = y_dias` y `y_modif = NA`. Se construye con `python -m src.base`.
- **Base v1** (domingo 25): pipeline completo.

### 2b. Función única de features

```python
# src/features.py  (dueño: Kevin; cambios por PR revisado por Jose y Geronimo)
FEATURES_VERSION = "1"
COLS_CONTRATO: list[str]   # columnas crudas de jbjy-vk9h que usa el modelo
COLS_PROCESO: list[str]    # columnas crudas de p6dx-8zbt, con prefijo p_
FEATURES_CAT, FEATURES_NUM: list[str]
FEATURES = FEATURES_CAT + FEATURES_NUM
COLUMNAS_FUGA: set[str]    # espejo ejecutable de docs/LISTA_FUGA.md

def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe filas con COLS_CONTRATO + COLS_PROCESO (texto como lo devuelve la API, o ya tipadas)
    y devuelve exactamente FEATURES, en orden, con dtypes fijos y el mismo índice.
    Fila a fila, idempotente, no lee columnas y* ni eval_*, no hace requests.
    Faltantes -> NaN (la imputación es del Pipeline)."""

# src/uniones.py  (dueño: Kevin)
def elegir_proceso(procesos: pd.DataFrame) -> pd.DataFrame:            # una fila por id_del_portafolio (4.4)
def unir_contrato_proceso(contratos, procesos) -> pd.DataFrame:         # left join, prefijo p_, falla si duplica
```

Se usa igual en los dos lados:
- Entrenamiento: `X = construir_features(base)`.
- App: `X = construir_features(unir_contrato_proceso(api.contrato(id), elegir_proceso(api.procesos(portafolio))))`.

Familias de features v1, todas disponibles al firmar:
- **Contrato:** tipo, modalidad y su justificación, UNSPSC por segmento y familia (el código es `V1.` + 8 dígitos, o `UNSPECIFIED` en el 2,2 %; el `unspsc_6` de la exploración eran solo 3 dígitos) [VERIFICADO], pago adelantado, origen y destino de los recursos, posconflicto, documentos tipo, tipo de documento del proveedor (solo el tipo), grupo/pyme, longitud del objeto.
- **Entidad y geografía:** NIT, sector, orden, rama, centralizada, departamento, ciudad.
- **Calendario (H4):** mes, día de la semana, último trimestre, fin planeado que cruza el 31 de diciembre, días de firma a inicio.
- **Proceso (H1, H2, H5):** log del precio base, duración planeada, días de publicación a firma, lotes, invitados, manifestaciones de interés, respuestas, proveedores únicos.
- **En el Pipeline de Jose, no aquí:** tasa histórica de la entidad (H6, `TargetEncoder`) y, si da el tiempo, duración frente a la típica del objeto (H5).

### 2c. Marco de evaluación

```python
# src/evaluacion.py  (dueño: Jose)
def split_temporal(base, anio_test, anio_inicio=2019) -> tuple[pd.Index, pd.Index]   # ambos con eval_madura
def folds_temporales(anios, anios_val=(2021, 2022, 2023)) -> list[tuple]   # ventana creciente; sirve como cv=
def regla_tasa_base(y_train, n) -> np.ndarray
def regla_tasa_entidad(ent_train, y_train, ent_test, m) -> np.ndarray     # (pos + m*p0)/(n + m); no vistas -> p0
def elegir_m(ent, y, anios) -> float                                      # validación temporal dentro del train; ROC-AUC y precisión en el top-k (sección 0, punto 11)
def metricas(y_true, score, fracciones_k=(0.01, 0.05, 0.10)) -> dict      # n, tasa_base, pr_auc, roc_auc, pr_auc/base, P@k, R@k
def comparar(y_true, scores: dict, entidad, referencia="regla_entidad", n_boot=200, seed=42) -> pd.DataFrame
    # métricas + diferencia contra la referencia con IC 95 % (bootstrap POR ENTIDAD)
def evaluar_por_grupo(y_true, score, grupo) -> pd.DataFrame               # duración planeada, entidad vista/no vista
def evaluar(estimador, base, anio_test) -> dict                           # {"scores", "tabla", "por_grupo"}
```

- `estimador` puede ser cualquier objeto con `fit`/`predict_proba` que reciba la salida de `construir_features`.
- **Criterio de aceptación:** en test 2024, el IC 95 % (bootstrap por entidad) de la diferencia de PR-AUC contra la regla por entidad excluye el cero.
- **Deriva 2025:** (i) **mismo modelo** 2019-2023 evaluado en 2025 [MÍNIMO; también alimenta 3.9]; (ii) modelo reentrenado 2019-2024 evaluado en 2025 [MÍNIMO, recortable].

### 2d. Artefacto del modelo (se versiona en `models/`)

```
models/modelo_prorroga_v1.joblib     # Pipeline completo; recibe construir_features(...)
models/modelo_prorroga_v1.json       # metadatos
models/modelo_prorroga_dummy.joblib  # HGB por defecto (sin early stopping) sobre la base v0, con las categorías como códigos numéricos
                                     # (sin categorías nativas, ver sección 0, punto 7), para Geronimo (+ su .json)
```

- **Versión:** basta con una sola, la v1. El versionado SemVer queda como [EXTRA]; la estrategia de reajuste (3.9) describe cómo se numeraría una v2.
- **Metadatos:** `version`, `creado`, `features_version`, `features`, `etiqueta`, `datos` (corte, años, n, tasa base, dígito), `modelo` (clase, hiperparámetros, semilla 42), `metricas` (2024, 2025, reglas), `umbrales_riesgo` (bajo/medio/alto por percentiles de validación), `entorno` (versiones de Python, scikit-learn, pandas y shap), `sha256`, `nombres_legibles` (feature original → texto en español).
- **Carga:** `src/modelo_io.py: cargar_modelo(ruta) -> (pipeline, meta)` falla con un mensaje claro si no coinciden `features_version` o la versión de scikit-learn.
- **Lo que muestra la app:** nivel de riesgo y percentil, no la probabilidad cruda (con la deriva queda descalibrada).

**Cómo se cambian los contratos:** solo por PR revisado por los dueños afectados, subiendo `FEATURES_VERSION` o la versión de la base. `tests/test_contratos.py` (pytest) tiene **solo tres verificaciones**:
1. El esquema de la base.
2. `FEATURES ∩ COLUMNAS_FUGA = ∅`.
3. `construir_features` da los mismos dtypes con entrada de texto (como la API) que con entrada tipada.

---

## 3. Fases, responsables y calendario

```
F0 (Kevin) ──┬─> J1 ─> J2 ─> J3 ───────────────────┐
 base v0     ├─> G1 ─> G2 (dummy) ─> G3 ─> app v1 ──┤
 + dummy     ├─> F4 EDA (v0 → v1) ──────────────────┤──> I (vie 30) ─> revisión y entrega (sáb 31)
             └─> F1 ─> F2 ─> F3 base v1 ────────────┘
```

### Calendario

| Fecha | Kevin | Jose | Geronimo |
|---|---|---|---|
| Jue 22 | **F0 completa**; base v0 y dummy compartidos esa noche | Setup y lectura (incluye leer los contratos de F0) | Setup y lectura (incluye documentación de Streamlit Cloud y SHAP) |
| Vie 23 | **F1** (máximo medio día para la etiqueta) | **J1** | **G1**, con la prueba de SHAP sobre el dummy |
| Sáb 24 - Dom 25 | **F2** (descarga de noche) + **F3** (base v1 el domingo) + inicio del EDA con v0 | J1 → **J2** | Termina **G1** |
| Lun 26 - Mar 27 | **EDA con v1** | J2 → **J3** | **G2** con el dummy → **G3** |
| Mié 28 - Jue 29 | Revisa PRs | **J3** (modelo elegido el jueves) | **G3** + modelo real en la app |
| Vie 30 | **README** | **Integración del notebook** | **Video** |
| Sáb 31 | **Revisión final y entrega** | Correcciones de la revisión | Verifica la app desplegada |
| Lun 2 nov | Ensayo de sustentación (todos) | | |
| Mar 3 nov | **Sustentación** | | |

**Checkpoints:**
- Etiqueta: viernes 23 por la noche.
- Base v1: domingo 25 por la noche.
- Modelo elegido: jueves 29.
- Notebook integrado: viernes 30 por la noche.

### Límites de tiempo (reglas duras)

| Tarea | Límite | Si se pasa |
|---|---|---|
| Verificación de la etiqueta (F1) | Medio día | Si no es concluyente: `y = dias_adicionados > 0`, se documenta la limitación y F2 no descarga Modificaciones |
| Descarga F2 | La estimación de F1 no puede pasar de **3 h** | Si el exceso viene de Modificaciones, se usa `y = y_dias`. Si viene de Contratos o Procesos (p. ej. `Crawl-delay`), se baja al 5 % (5 de las 100 terminaciones de dos dígitos) |
| Búsqueda de hiperparámetros (J3) | **15 min por modelo** | Se mide un ajuste y se fija `n_iter` para no pasar el tope. kNN y SVM sobre una submuestra pequeña (~20 mil de train) |
| `tests/test_contratos.py` | Solo las 3 verificaciones de 2d | — |

### Estimaciones recalculadas [ESTIMADO]

Supuesto: jornadas de ~8 h efectivas por persona y por día, fines de semana incluidos [SUPUESTO].

| Fase | Responsable | Esfuerzo | Ventana | ¿Cabe? |
|---|---|---|---|---|
| F0 | Kevin | 7-8 h | Jue 22 (8 h) | **Justo** |
| F1 | Kevin | 6-7 h (etiqueta ≤ 4 h) | Vie 23 (8 h) | Sí |
| F2 | Kevin | 4-5 h de código + 20-50 min de descarga (tope 3 h) | Sáb 24 | Sí, si la API responde como en la exploración |
| F3 | Kevin | 5-6 h | Dom 25 | **Justo**: el fin de semana suma ~15 h de Kevin |
| F4 (EDA) | Kevin | 12-14 h (~4 h el fin de semana con v0 + ~9 h con v1) | Sáb 24 - Mar 27 | **Justo** |
| Revisión de PRs | Kevin | ~1 h por PR (~6 PRs) + 2-3 h el notebook | Continua; vie 30 por la noche y sáb 31 | Sí |
| README | Kevin | 3 h | Vie 30 | Sí |
| Setup y lectura | Jose, Geronimo | 3-5 h | Jue 22 | Sí |
| J1 | Jose | 8-9 h | Vie 23 - Sáb 24 | Sí |
| J2 | Jose | 6-8 h | Sáb 24 - Lun 26 | Sí |
| J3 | Jose | 18-20 h (cómputo ≤ 5 × 15 min por corrida de búsqueda) | Lun 26 - Jue 29 (~3 días) | Sí, con los topes |
| I (integración) | Jose | 6-8 h, incluida una ejecución completa con `nbconvert` (meta ≤ 20 min) | Vie 30 (1 día) | **Sin holgura** |
| G1 | Geronimo | 12-14 h | Vie 23 - Dom 25 | Sí |
| G2 | Geronimo | 3-4 h con el dummy + 1-2 h con el modelo real | Lun 26; Jue 29 | Sí |
| G3 | Geronimo | 10-12 h | Lun 26 - Jue 29 | Sí: la simulación usa el `Pipeline` de J2 con un HGB por defecto y no espera a J3 |
| Video | Geronimo | 3-4 h | Vie 30 | Sí, si el jueves la app ya tiene el modelo real |

### Fase en riesgo

**La de mayor riesgo es F2 + F3, el fin de semana de Kevin.**
- **Por qué:** es el único tramo del camino crítico que depende de algo externo (la velocidad de la API, el `Crawl-delay` y Modificaciones con 30 M de filas) y concentra ~15 h de Kevin junto con el inicio del EDA. Si la base v1 se corre al martes, J3 pierde un día de sus ~3, el EDA con v1 queda en un solo día y el modelo real llega tarde al video.
- **Mitigaciones:**
  - Dejar `descarga.py` probado el viernes por la tarde con un mes, después de la media jornada de la etiqueta.
  - Lanzar Contratos y Procesos el sábado temprano (no dependen de la etiqueta) y Modificaciones el sábado por la noche.
  - Escribir el EDA como funciones sobre v0 para re-ejecutarlo en v1 sin rehacerlo.
  - **Corte duro:** si el domingo 25 a las 22:00 la base v1 no está completa, se publica una v1 con `y = y_dias` (solo Contratos + Procesos) y Modificaciones queda fuera.

**Segunda en riesgo: la integración del viernes 30 (Jose).** Tiene un solo día y depende de que todo lo anterior haya llegado.
- **Mitigaciones:**
  - Cada dueño entrega su borrador ya en formato final (celdas markdown + código que llama a `src/`) el **jueves 29 por la noche**.
  - Jose crea el esqueleto del notebook (secciones 3.1-3.9, imports y carga de datos) el jueves.
  - El notebook final lee los resultados guardados de la búsqueda (`resultados/*.csv`, versionados) y reentrena solo el modelo elegido. Un flag `RECALCULAR = False` permite repetir la búsqueda. Meta: ejecución completa en ≤ 20 min.
  - Kevin revisa el viernes por la noche, no solo el sábado.

### Kevin

**F0. Cimientos y contratos (jue 22)**
- Rama: `feat/cimientos` (una sola para toda la F0, con commits separados por paso).
- Tareas [MÍNIMO]:
  - Ubicar el venv del repo (si no existe, preguntarte antes de crearlo). No existía: se creó `final-project/.venv` con Python 3.12.10.
  - `requirements.txt` con versiones fijas, incluidos `streamlit`, `shap` y `pytest` (autorizados); ver la sección 0, punto 8.
  - `final-project/.gitignore` con `data/`, `.env` y `**/.streamlit/secrets.toml`, y `final-project/.gitattributes` con `* text=auto`.
  - `src/config.py`: IDs de los datasets, semilla y **lista PII ampliada**.
  - `src/features.py` y `src/uniones.py`, portados de la exploración sin `visualizaciones_del`.
  - Base v0, modelo dummy y `tests/test_contratos.py` (las 3 verificaciones).
  - Subir la base v0 y el dummy a Drive/OneDrive con sha256 esa noche.
- Los PRs de cimientos y documentación de Kevin (F0 y docs) se mergean con la aprobación general del equipo, sin revisión formal (excepción en `REGLAS_DEL_PROYECTO.md`). Jose y Geronimo leen los contratos el jueves por la noche.
- **Hecho en la F0:** base v0 de 167.997 contratos (2019-2025; la muestra de la exploración trae 24.000 por año de firma, y 2018 y 2026 se descartan), dummy, `cargar_modelo` y `tests/test_contratos.py`. El dummy es un HGB con los hiperparámetros por defecto salvo `early_stopping=False` (con `auto` valida con una partición aleatoria, lo que prohíbe la regla 4), entrenado con 2019-2023 maduros. En 2024 da 0,766 ROC-AUC / 0,344 PR-AUC; solo es una verificación de cordura y sus cifras no se citan.
- Listo cuando: Jose obtiene `FEATURES` con la base v0, Geronimo predice con el dummy y `pytest` pasa.

**F1. Verificación previa (vie 23)**
- Rama: `data/verificacion-etiqueta`.
- Tareas [MÍNIMO]:
  - **Prioridad 1: verificar `fecha_de_inicio_del_contrato`** (de ella dependen `dias_firma_a_inicio` y `fin_planeado_cruza_anio`, que explican la ventaja del modelo sobre la regla; sección 0, punto 9). En la muestra de verificación de Modificaciones (4.3), comparar `fecha_inicio_contrato` de la primera versión con el valor actual de `fecha_de_inicio_del_contrato` en Contratos. Reportar el % de contratos en que cambia, separado por `y = 1` frente a `y = 0`. Si cambia más en `y = 1`, hay posible fuga y el equipo decide si esas dos features salen de `FEATURES`.
  - Leer el `robots.txt` completo de datos.gov.co, en un paso separado.
  - Verificación de la etiqueta (4.3): **máximo medio día**.
  - Contar contratos por último dígito y año.
  - Medir 20 lotes de Procesos y 20 de Modificaciones, y **estimar F2 contra el tope de 3 h**.
  - Revisar las columnas "por verificar" con los datos de la exploración.
  - Por la tarde: `descarga.py` probado con un mes.
- Listo cuando: el equipo aprueba la etiqueta el viernes por la noche.

**F2. Descarga (sáb 24, de noche)**
- Rama: `feat/descarga-secop`.
- Tareas [MÍNIMO]: `src/descarga.py` (4.6). Reutiliza `soql`, `in_list`, `Throttled` y `por_lotes` de `00_exploracion/secop_descarga.py`, con timeout de 30 s, 3 reintentos y App Token.
- Contratos y Procesos se lanzan temprano; Modificaciones, de noche.
- Listo cuando: el manifiesto no tiene lotes pendientes y el conteo por año cuadra con el 10 % de la población (±2 %).

**F3. Base v1 y calidad (dom 25)**
- Rama: `data/base-contratos-v1`.
- Tareas [MÍNIMO]: `src/etiqueta.py`, censura (4.5), `src/calidad.py` (4.7) y `docs/LISTA_FUGA.md`.
- [EXTRA] Detector de columnas mutables.
- Listo cuando: la calidad pasa y la base v1 está compartida con su sha256 el domingo por la noche (corte duro a las 22:00).

**F4. EDA (sáb 24 - mar 27)**
- Rama: `feat/eda`, en el borrador `notebooks/03_eda.ipynb`.
- Tareas [MÍNIMO]:
  - **H1-H7 con evidencia** (tasa de prórroga por grupo con intervalo, gráfica e interpretación): responde al profesor cómo los datos de la firma predicen la prórroga.
  - Etiqueta `y` frente a `y_dias` por año.
  - Censura y duración; deriva; evidencia de fuga (contraste 0,95 / 0,84).
  - **Decisión de privacidad del proveedor** (abajo).
- Listo cuando: el borrador está en formato final el martes 27 y cada insight está enlazado a una decisión (Decisión / Alternativas / Por qué / Evidencia).

> **Decisión de privacidad del proveedor (F4, la cita J3):**
> - **Decisión:** excluir la identidad y los datos del proveedor (documento, nombre, `codigo_proveedor`, NIT adjudicado).
> - **Alternativas:** historial de prórrogas del proveedor; identificador cifrado (hash); usarlo solo para personas jurídicas (NIT); excluir.
> - **Por qué:** muchos proveedores son personas naturales (prestación de servicios) y su identificador es un dato personal (Ley 1581 de 2012). El hash sigue siendo un dato personal seudonimizado, y un puntaje por proveedor convertiría la app en un ranking de personas.
> - **Evidencia:** distribución de `tipodocproveedor` (% cédula frente a NIT).
> - **Costo reconocido:** se pierde una posible señal (el historial del proveedor) cuyo tamaño no se midió [NO EVALUADO]. Se conserva `tipodocproveedor` (el tipo, no la identidad).

**Además:** revisión de PRs el mismo día en que llegan (concentrada el mié 28 y el jue 29); README el vie 30; revisión del notebook integrado el vie 30 por la noche; revisión final y entrega el sáb 31.

### Jose

**J1. Marco de evaluación (vie 23 - sáb 24)**
- Rama: `feat/marco-evaluacion`.
- Tareas [MÍNIMO]:
  - Implementar 2c con pruebas mínimas.
  - `elegir_m` considera ROC-AUC y precisión en el top-k, y se reporta la regla suavizada junto a la de m=0 (sección 0, punto 11): con m=0 la regla es un rival débil en el top 1 %.
  - Reportar el desempeño en entidades no vistas y la precisión y el recall en el top 1, 5 y 10 % (referencias en la sección 0, punto 11).
- Listo cuando: con la base v0, la regla por entidad da ≈ 0,714 / 0,277 en 2024 (±0,005, cifras de la exploración) y el HGB sobre `FEATURES` da ≈ 0,751 / 0,333 (referencia nueva medida en la F0, sección 0, punto 9). Ese montaje usa categorías nativas sobre un `OrdinalEncoder` de las 200 categorías más frecuentes, `random_state=42` y el resto por defecto (sección 0, punto 7). El HGB A original (0,763 / 0,334) incluía los 6 montos y ya no es la referencia.

**J2. Preprocesamiento (sáb 24 - lun 26)**
- Rama: `feat/preprocesamiento`.
- Tareas [MÍNIMO]: `ColumnTransformer` por familia de modelo (categorías nativas para HGB; one-hot + escalado para kNN y SVM; `TargetEncoder` para la entidad), faltantes y atípicos, todo justificado. Para el HGB deja listas las dos variantes (con y sin categorías nativas), porque J3 las compara por su efecto en SHAP (sección 0, punto 7).
- Listo cuando: todo vive en el `Pipeline` y se ajusta solo con train. Ese `Pipeline` es el que Geronimo usa en G3.

**J3. Modelado y selección (lun 26 - jue 29)**
- Rama: `feat/modelado`.
- Tareas [MÍNIMO]:
  - kNN, árbol, RF, HGB y SVM, con búsqueda sobre `folds_temporales` y **tope de 15 min por modelo**; kNN y SVM sobre submuestra.
  - Early stopping desactivado o temporal.
  - Variante C sin `visualizaciones_del`.
  - Test 2024 una sola vez; deriva 2025 (i) y, si da el tiempo, (ii).
  - **SHAP global enlazado a H1-H7**; sensibilidad `y` frente a `y_dias`.
  - **Decisión sobre categorías nativas y SHAP (sección 0, punto 7):** medir el PR-AUC del HGB con y sin categorías nativas y elegir (a) o (b). Evidencia previa de la F0: los códigos ordinales no pierden (0,766 / 0,344 frente a 0,752 / 0,336 en los maduros de 2024; una corrida, sin intervalo; sección 0, punto 11). El artefacto v1 y la app usan el mismo explicador, y se verifica la aditividad de SHAP en el modelo final.
  - Resultados de la búsqueda guardados en `resultados/`; artefacto v1.
  - Decidir si el modelo de producción se reentrena con 2019-2025.
- Tareas [EXTRA]: complemento no supervisado del 3.4; curva de aprendizaje para decidir la ampliación al 20 %.
- Listo cuando: el modelo cumple el criterio de 2c y el artefacto pasa `cargar_modelo` el jueves 29.

**I. Integración (vie 30)**
- Rama: `docs/notebook-final`.
- Tareas [MÍNIMO]:
  - El jueves 29: esqueleto del notebook y borradores de todos recibidos.
  - El viernes 30: armar `model_development.ipynb` (§3.1-3.9, en orden) y ejecutarlo con `jupyter nbconvert --execute` (≤ 20 min).
  - Kevin revisa el viernes por la noche; correcciones el sábado 31.
- Listo cuando: corre de principio a fin sin errores y cada decisión está en el formato.

### Geronimo

**G1. App con el dummy (vie 23 - dom 25)**
- Rama: `feat/app-consulta`.
- Tareas [MÍNIMO]:
  - `src/api_secop.py`: contrato y procesos; App Token desde `.env` o `st.secrets`; timeout de 30 s, 3 reintentos y caché. Verificar qué identificador tiene de verdad el usuario.
  - `app/app.py`: riesgo, percentil, factores en lenguaje claro y avisos (no detecta corrupción, entidad sin historial, fuera del periodo, sin proceso).
  - **Prueba de SHAP con el dummy sobre el Pipeline (vie 23):**
    - `TreeExplainer` sobre el estimador final, con `pipeline[:-1].transform(X)` y `get_feature_names_out()`.
    - Agregar las columnas transformadas a la feature original y traducirlas con `nombres_legibles`.
    - **Verificar la aditividad, no solo que corra:** la suma de los valores SHAP más el valor base debe igualar la salida que se explica (log-odds con `decision_function` en `TreeExplainer`), con error máximo < 1e-3. [VERIFICADO con datos sintéticos] `TreeExplainer` sobre un HGB con categorías nativas no lanza error pero da atribuciones incorrectas (error hasta 13,6 log-odds); sin categorías nativas es exacto. El dummy (HGB por defecto sobre códigos numéricos) no está afectado.
    - Si el modelo final usa categorías nativas: explicar con `shap.Explainer` genérico sobre `predict_proba` del Pipeline (funcionó en la prueba sintética; más lento) y repetir la prueba de aditividad.
    - **Par fijado en la F0:** `scikit-learn` 1.9.0 + `shap` 0.54.1 (sección 0, punto 8). Si G1 encuentra un problema con el par, se avisa antes del sábado 24, antes de que nadie entrene modelos que se guarden.
  - Prueba de paridad API frente a base con 20 contratos (script, no test automático).
- Listo cuando: la app corre en local con 5 contratos reales y muestra 3-5 factores en español.

**G2. Despliegue (lun 26 con el dummy; jue 29 con el modelo real)**
- Rama: `feat/despliegue`.
- Tareas [MÍNIMO]: medir la RAM en local, desplegar (sección 5) y cambiar al modelo v1 el jueves.
- [EXTRA] Consulta por entidad.
- Listo cuando: la URL pública responde de punta a punta desde otro equipo.

**G3. Reajuste (lun 26 - jue 29)**
- Rama: `feat/reajuste`.
- Tareas [MÍNIMO]:
  - `src/monitoreo.py`: PR-AUC de los contratos que van madurando, PSI del puntaje y de las features clave, tasa base, criterio de alarma y de reentrenamiento, validación campeón/retador, y cómo se pondría en producción una v2.
  - Simulación con el `Pipeline` de J2 + HGB por defecto entrenado en 2019-2023, recorriendo 2024-2025 mes a mes (o por trimestre si se aprieta), para mostrar dónde salta la alarma.
- Listo cuando: la simulación corre, los umbrales están justificados y el borrador de §3.8-3.9 está listo el jueves 29.

**Video (vie 30):** se graba con la app desplegada y el modelo v1.

### Recortes [EXTRA] (fuera del alcance salvo que sobre tiempo)

1. Complemento no supervisado del 3.4.
2. Consulta por entidad.
3. Detector de columnas mutables.
4. Ampliación al 20 %.
5. Versionado SemVer (basta con v1).

**Si aun así falta tiempo**, se reduce el alcance de tareas [MÍNIMO] en este orden, sin eliminarlas:
1. Deriva (ii): queda solo la (i).
2. G3: de mensual a trimestral.
3. Rejillas más chicas en kNN y SVM.

La etiqueta y la descarga ya tienen su salida en los límites de tiempo.

**Nunca se recorta:** split temporal, comparación con las reglas simples, lista de fuga, formato Decisión / Alternativas / Por qué / Evidencia, H1-H7 con evidencia, app desplegada, video y notebook ejecutado.

**Checklist de revisión de PR (Kevin):**
- No rompe contratos, o sube la versión.
- Ninguna feature en `COLUMNAS_FUGA`.
- Split y validación temporales; todo lo aprendido dentro del `Pipeline`; semilla 42.
- Sin datos ni tokens en el diff: **el repo es público**.
- Decisiones en el formato del notebook.

---

## 4. Pipeline de datos (detalle de la parte de Kevin)

### 4.1 Volumen y RAM: decidido, 10 % por dígito

- **Muestra:** aleatoria del 10 % por último dígito de `id_contrato`. Es proporcional por año y reproducible.
- **Tamaño:** ~450 mil contratos 2019-2025 (train ~260 mil, test 2024 ~88 mil, test 2025 ~102 mil).
- **Costo:** descarga de 20-50 min (tope de 3 h); pico de RAM < 2-3 GB procesando por meses; base v1 de ~150-300 MB en memoria [ESTIMADO].
- **Control de RAM:**
  - Procesar mes a mes y solo con las columnas necesarias.
  - Tipos explícitos (category, float32, Int16, `string[pyarrow]`).
  - Nada de `dtype=str` para la tabla entera.
  - Registrar el pico con `psutil`.
- **Otras opciones:** el 20 % es [EXTRA]; el 5 % es la salida si F2 pasa de 3 h; la población completa se descarta.

### 4.2 Columnas a descargar (verificadas contra `meta/*.json`) [VERIFICADO]

**Contratos (`jbjy-vk9h`):**
- Llaves y momento: `id_contrato`, `proceso_de_compra`, `referencia_del_contrato`, `fecha_de_firma`.
- Entidad: `nombre_entidad`, `nit_entidad`, `departamento`, `ciudad`, `orden`, `sector`, `rama`, `entidad_centralizada`.
- Diseño: `tipo_de_contrato`, `modalidad_de_contratacion`, `justificacion_modalidad_de`, `codigo_de_categoria_principal`, `objeto_del_contrato`, `es_grupo`, `es_pyme`, `tipodocproveedor`, `habilita_pago_adelantado`, `valor_de_pago_adelantado`, `origen_de_los_recursos`, `destino_gasto`, `espostconflicto`, `documentos_tipo`, `el_contrato_puede_ser_prorrogado` (por verificar; excluida de `FEATURES`, sección 0, punto 10).
- Fuentes de recursos (fuga confirmada; solo en `data/raw/` para el EDA de fuga, nunca en la base): `presupuesto_general_de_la_nacion_pgn`, `sistema_general_de_participaciones`, `sistema_general_de_regal_as`, `recursos_propios_alcald_as_gobernaciones_y_resguardos_ind_genas_`, `recursos_de_credito`, `recursos_propios`.
- Fechas (por verificar): `fecha_de_inicio_del_contrato`, `duraci_n_del_contrato`.
- Etiqueta: `dias_adicionados`.
- Solo en `data/raw/` (EDA de fuga, nunca en la base): `fecha_de_fin_del_contrato`, `estado_contrato`, `valor_del_contrato`.
- **No se descargan:** las 21 PII de `secop_descarga.py` + `codigo_proveedor` (PII ampliada); pagos, saldos, liquidación, reversión, `ultima_actualizacion`, `fecha_de_notificaci_n_de_prorrogaci_n`.

**Procesos (`p6dx-8zbt`):**
- Deduplicación: `id_del_portafolio`, `id_del_proceso`, `adjudicado`, `fecha_adjudicacion`, `valor_total_adjudicacion`.
- Tiempos (H1): `fecha_de_publicacion_del`, `fecha_de_recepcion_de`, `fecha_de_apertura_efectiva`.
- Plan: `precio_base`, `duracion`, `unidad_de_duracion`, `numero_de_lotes`, `subtipo_de_contrato`.
- Competencia (H2, por verificar): `proveedores_invitados`, `proveedores_con_invitacion`, `proveedores_que_manifestaron`, `respuestas_al_procedimiento`, `respuestas_externas`, `conteo_de_respuestas_a_ofertas`, `proveedores_unicos_con`.
- **No se descargan:** `visualizaciones_del`, `estado_del_procedimiento`, `id_estado_del_procedimiento`, `estado_resumen`, `fase`, `estado_de_apertura_del_proceso`, `fecha_de_ultima_publicaci`, y por PII ampliada `nombre_del_adjudicador`, `nombre_del_proveedor`, `nit_del_proveedor_adjudicado`, `codigoproveedor`.

**Modificaciones (`u8cx-r425`):**
- Verificación: `id_contrato`, `identificador`, `numero_version`, `estado_modificacion`, `dias_extendidos`, `fecha_creacion`, `fecha_de_aprobacion`, `fecha_version`, `fecha_inicio_contrato`, `fecha_fin_contrato`, `proposito_modificacion`, `descripcion`, `valor_modificacion`, `sustituto_version`.
- Etiqueta (filtro en el servidor `dias_extendidos > 0`): `id_contrato`, `numero_version`, `estado_modificacion`, `dias_extendidos`, `fecha_de_aprobacion`, `fecha_fin_contrato`.
- No se descarga `autor_version` (PII ampliada).

### 4.3 Etiqueta (máximo medio día el viernes 23)

**Verificación** (< 1 min de descarga). Unos 600 contratos de la base v0: ~200 con `y_dias = 1`, ~200 discordantes ya identificados en la exploración y ~200 al azar. Preguntas, en orden de prioridad:
1. ¿`dias_extendidos` es incremental o acumulado? Se compara con el cambio de `fecha_fin_contrato` entre versiones; define si se agrega con `max` o con `sum`.
2. ¿Qué estados de `estado_modificacion` son efectivos?
3. ¿Hay modificaciones aprobadas antes de la firma o el mismo día? Esas no cuentan.
4. Prórroga frente a suspensión, según `proposito_modificacion`.
5. Confirmar que todo `y_dias = 1` tiene su modificación (en la exploración, 347 de 347).

**Definición si es concluyente:** `y = 1` si `dias_adicionados > 0` **o** existe una versión efectiva con `dias_extendidos > 0` y `fecha_de_aprobacion > fecha_de_firma`.
- Efecto esperado: la tasa pasaría de ~12 % a ~25-30 % en 2024 [ESTIMADO con 844 / 347 de la submuestra, posiblemente sesgado].
- Hipótesis para el EDA: si la tasa combinada es estable entre años, el salto de 2025 sería un cambio de registro [SUPUESTO].

**Si no es concluyente al mediodía:** `y = dias_adicionados > 0`; la subestimación (59 % en la submuestra) se documenta como limitación en §3.3 y §3.6, y F2 no descarga Modificaciones.

**Costo de la etiqueta combinada:** ~3.000 lotes de 150 IDs, ~150-300 mil filas filtradas, 8-35 min con 3 hilos [ESTIMADO; se mide en F1].

### 4.4 Deduplicación de Procesos y verificación del join

- **Llave:** `proceso_de_compra` = `id_del_portafolio` [VERIFICADO en la exploración].
- **Regla** (`elegir_proceso`): dentro de cada portafolio se ordena por `adjudicado = 'Si'` primero, luego `fecha_adjudicacion` más reciente y luego el sufijo numérico más alto de `id_del_proceso`, y se toma la primera fila. Se agregan `p_n_filas` y `p_valor_adjudicado_max`.
- **Asserts:**
  - El portafolio es único tras la regla.
  - Las filas antes y después del join son iguales, y `id_contrato` es único.
  - Al menos el 95 % de los contratos tiene proceso (la exploración dio 99,8 %).
- **Se reportan:** los portafolios ambiguos.
- **En la app:** usa la misma función.

### 4.5 Censura

- No se usa `fecha_de_fin_del_contrato`, porque depende de la etiqueta.
- `eval_fin_planeado` = inicio (o la firma, si falta) + duración planeada. La duración sale del proceso; si falta, de `duraci_n_del_contrato`, siempre que F1 confirme que no se actualiza. En la v0 no hay respaldo: si falta la duración del proceso (0,26 % de los contratos), `eval_madura` queda en `False`.
- `eval_madura` = fin planeado + 60 días ≤ fecha de corte. La prórroga debe firmarse antes del vencimiento [SUPUESTO legal].
- Train y test solo con contratos maduros.
- Se reportan:
  - la fracción excluida por año;
  - las métricas por tramo de duración (≤ 3, 3-6, 6-12 y > 12 meses).

### 4.6 Almacenamiento, caché, reanudación y token

```
data/raw/        contratos/anio=AAAA/mes=MM_dig=D.parquet · procesos/lote_<sha1>.parquet
                 modificaciones/lote_<sha1>.parquet · _manifiesto.csv · meta/
data/processed/  base_contratos.parquet · base_contratos.meta.json · reporte_calidad.md
```

- **Reanudación:** un archivo por unidad de trabajo, escrito en `.tmp` y renombrado al terminar. Si existe, está completo; al reiniciar se salta.
- **Nombres deterministas:** cada lote se nombra por el hash de sus IDs ordenados. Las respuestas vacías se guardan como parquet vacío con esquema.
- **Manifiesto:** dataset, clave, filas, bytes, segundos, fecha, estado y sha256.
- **Requests:**
  - Timeout de 30 s, 3 reintentos con espera creciente y parada ante 403/429 (`Throttled`).
  - ≤ 3 hilos, o secuencial con 1 s si el `robots.txt` lo exige.
  - Paginación estable por `id_contrato`.
  - Log ASCII con lote i/N, filas, tiempo transcurrido, ETA y RAM.
- **App Token:**
  - Variable `SOCRATA_APP_TOKEN`. En local se lee de `final-project/.env` con un lector simple de líneas (sin `python-dotenv`); en la app, de `st.secrets`.
  - Se envía en el header `X-App-Token` [SUPUESTO; se confirma con la documentación de Socrata].
  - El token no se comparte por chat ni se sube al repo (que es público): Kevin crea el `.env` y el código solo lee la variable.

### 4.7 Chequeos de calidad (`src/calidad.py`, fallan con error)

1. `id_contrato` único; el join no duplica; ninguna columna PII; `FEATURES ∩ COLUMNAS_FUGA = ∅`.
2. Tamaño por año = 10 % de la población (±2 %), con `meta/poblacion_contratos_por_anio.csv`.
3. Tasa de `y_dias` por año en la muestra frente a la población (±1,5 pp).
4. Rangos de fechas (años 0025 y 8201 → NaT), montos ≥ 0, duración entre 1 y 3.650 días.
5. ≥ 95 % de los contratos con proceso; faltantes por feature; dtypes iguales al esquema.
6. Fracción madura por año; `y` frente a `y_dias` por año.
7. Ninguna feature con ROC-AUC univariado > 0,85 sin revisión.

### 4.8 `docs/LISTA_FUGA.md`

Columnas de la tabla: **Columna o derivado | Conjunto | Por qué es fuga | Evidencia | Estado (confirmada / por verificar / segura) | Uso permitido (ninguno / solo etiqueta / solo censura / solo EDA)**. `COLUMNAS_FUGA` es su espejo ejecutable.

| Columna | Estado | Evidencia o motivo | Uso |
|---|---|---|---|
| `estado_contrato` | Confirmada | AUC 0,94 | Solo EDA |
| `fecha_de_fin_del_contrato` y derivados | Confirmada | Razón de duración 1,44 frente a 1,00; AUC 0,71 | Solo EDA |
| `valor_del_contrato` | Confirmada | Razón de valor 1,18 frente a 1,00 | Solo EDA |
| Pagos, facturado, amortizado, pendientes, saldos (8 columnas) | Confirmada | Ejecución posterior (pendientes: AUC 0,63) | Ninguno |
| `ultima_actualizacion` | Confirmada | AUC 0,74 | Ninguno |
| Liquidación y reversión (6 columnas) | Confirmada | Posteriores al contrato | Ninguno |
| `fecha_de_notificaci_n_de_prorrogaci_n` | Confirmada | Registra la propia prórroga | Ninguno |
| `dias_adicionados`; dataset Modificaciones | Etiqueta | Definen `y` | Solo etiqueta |
| Procesos: visualizaciones, estados, fase, última publicación | Confirmada (reglas) | Estados actuales o acumulados | Ninguno |
| Ejecución y Adiciones (datasets) | Confirmada | Posteriores a la firma | Ninguno |
| 6 montos de fuentes de recursos | Confirmada | Su suma es igual a `valor_del_contrato` en el 80,6 %; cociente sobre el precio base de 1,26 con adición en el valor frente a 1,00 | Solo EDA |
| `fecha_de_inicio_del_contrato`, `duraci_n_del_contrato` | Por verificar | ¿Se actualizan? Prioridad 1 de F1: % que cambia frente a la primera versión de Modificaciones, por `y` | Pendiente |
| `el_contrato_puede_ser_prorrogado` | Por verificar, excluida | ¿Se edita después de la firma? ROC-AUC univariado 0,561 (sección 0, punto 10) | Solo ablación |
| Conteos de competencia, `valor_total_adjudicacion` | Por verificar | ¿Cambian después de la firma? | Pendiente |

---

## 5. Despliegue (la decisión es de Geronimo en G2)

Los límites son de memoria y **[SUPUESTO]**; Geronimo los confirma con la documentación oficial (ctx7) el jueves 22. El repo es público (`kepabonsaeafit/fundamentos-aa-2026`), lo que encaja con Streamlit Community Cloud.

| Opción | Límites gratis [SUPUESTO] | A favor | En contra |
|---|---|---|---|
| **Streamlit Community Cloud** | ~1 GB de RAM; se duerme tras días sin visitas; secrets | Despliega desde el mismo repo público, se redespliega en cada push, todo en Python | Se duerme (despertarla antes del video y de la sustentación); RAM justa si SHAP pesa |
| **Hugging Face Spaces** (CPU básico) | ~2 vCPU, ~16 GB de RAM, se duerme tras ~48 h | RAM holgada para SHAP | Repo aparte que hay que sincronizar; otra cuenta; Streamlit pudo pasar a Docker |
| **Render** (gratis) | ~512 MB de RAM; se duerme tras ~15 min | URL estable | RAM muy justa; arranque en frío de ~1 min en plena demo |

**Recomendación:** Streamlit Community Cloud, con HF Spaces como plan B si la RAM medida pasa de ~700 MB. Sea cual sea:
- `app/requirements.txt` con las mismas versiones del entrenamiento;
- token en los secrets;
- caché de la API;
- plan C local (`streamlit run`) para la sustentación.

---

## 6. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| **Disponibilidad del equipo** | Supuesto de 8 h/día por persona sin confirmar; Kevin confirma la disponibilidad real del equipo el miércoles 21. Si es menor, se aplica el orden de recortes de la sección 3. |
| **Fin de semana de Kevin (F2 + F3): camino crítico con dependencia externa** | `descarga.py` probado el viernes; Contratos y Procesos temprano el sábado; corte duro el domingo a las 22:00 con `y = y_dias`; tope de 3 h |
| **Integración en un solo día (vie 30)** | Borradores en formato final el jueves por la noche; esqueleto del notebook el jueves; resultados de la búsqueda guardados; ejecución ≤ 20 min; revisión de Kevin el viernes por la noche |
| **SHAP da atribuciones incorrectas sin avisar:** `TreeExplainer` con las categorías nativas del HGB (error de aditividad hasta 13,6 log-odds) [VERIFICADO con datos sintéticos] | Prueba de aditividad obligatoria en G1 y en el modelo final; decisión (a) `shap.Explainer` genérico o (b) sin categorías nativas, medida en J3; par fijado en la F0 (`scikit-learn` 1.9.0 + `shap` 0.54.1); los metadatos guardan las versiones |
| Límite de uso de la API | App Token; `Throttled`; caché reanudable; ≤ 3 hilos |
| `Crawl-delay: 1` aplica a la API | Se lee en F1; si F2 pasa de 3 h, se baja al 5 % |
| Etiqueta no concluyente | Medio día como máximo; si no alcanza, `y_dias` con la limitación documentada |
| RAM en laptops | Procesar por mes, tipar, submuestra para kNN y SVM, `psutil` |
| Censura | Fin planeado, solo contratos maduros, métricas por duración |
| Fuga sutil | Columnas "por verificar", alarma de AUC univariado, variante C recalculada |
| **La ventaja del modelo limpio sobre la regla por entidad es modesta** (ROC-AUC +0,037 [+0,011, +0,069]; PR-AUC +0,056 [+0,003, +0,113]) y depende de 3 features que usan `fecha_de_inicio_del_contrato`, que está por verificar. | Verificar esa fecha en F1 (prioridad 1); medir el desempeño en entidades no vistas y la precisión en los primeros *k*; etiqueta mejorada, muestra del 10 % y ajuste de hiperparámetros en J3. Si al final no gana con claridad, se reporta con honestidad y se evalúa el valor del modelo en entidades no vistas y su combinación con la regla. Medido en la F0 (sección 0, punto 11): ventaja clara en entidades no vistas (ROC-AUC +0,208 [+0,084, +0,300]) y, en el top-k, solo en el 1 % más alto frente a la regla suavizada. |
| Entidades no vistas | `TargetEncoder` suavizado; IC por entidad; aviso en la app |
| Deriva de 2025 | Test aparte; riesgo en percentil; G3 la usa como alarma |
| Diferencias entre entrenamiento y app | Una sola `construir_features`; prueba de dtypes y paridad con 20 contratos |
| Modelo real tarde para el video | La app ya funciona con el dummy desde el lun 26; el cambio es un archivo |
| Repo público | Nunca datos, tokens ni PII; `.env` y secrets en `.gitignore`; la revisión de PR lo verifica |
| App dormida en la demo | Despertarla antes; plan C local |
| El riesgo se lee como acusación | Textos de la app según §8 de la definición |

---

## 7. Decisiones tomadas (2026-10-21) y cambios a las reglas

| # | Decisión |
|---|---|
| 1 | Entrega el domingo 1 de noviembre (meta interna: sábado 31); ensayo el lunes 2; sustentación el martes 3 |
| 2 | Muestra del 10 % por dígito |
| 3 | App Token: Kevin crea la cuenta; va en `.env` (gitignored) y en los secrets |
| 4 | Datos compartidos en Drive/OneDrive con sha256 |
| 5 | El artefacto se versiona en `models/`, solo la v1 |
| 6 | `streamlit`, `shap` y `pytest` autorizados, con versiones fijas |
| 7 | Profesor: tema aprobado; pide justificar cómo los datos del contrato predicen la prórroga y qué ingresa el usuario. Las demás decisiones las toma el equipo y se documentan en el notebook |
| 8 | Repo público: github.com/kepabonsaeafit/fundamentos-aa-2026 |
| 9 | Consulta por entidad, no supervisado, columnas mutables, 20 % y SemVer: [EXTRA] |
| 10 | Commit aparte con `docs/`; `.gitignore` dentro de `final-project/`; rama base `master` (se corrige en las reglas) |
| 11 | PII ampliada: `codigo_proveedor`, `nombre_del_adjudicador`, `nombre_del_proveedor`, `nit_del_proveedor_adjudicado`, `codigoproveedor`, `autor_version` |

**Cambios a las reglas que propone este plan** (los hace Kevin en el PR de F0, con el acuerdo del equipo):
- El App Token va en `.env` y en los secrets.
- `models/` se versiona.
- La lista PII pasa a `src/config.py`.
- Los borradores del notebook van en `notebooks/`.

---

## 8. Verificación

| Día | Qué se verifica |
|---|---|
| Jue 22 | `pytest tests/test_contratos.py` pasa con la base v0 en las tres laptops |
| Vie 23 | SHAP muestra 3-5 factores legibles con el dummy y la aditividad cumple (error máximo < 1e-3) |
| Sáb 24 | J1 reproduce las cifras de la exploración con la base v0 |
| Dom 25 | `calidad.py` sin errores con la base v1 |
| Jue 29 | El artefacto v1 pasa `cargar_modelo` y la app desplegada lo usa |
| Vie 30 | `jupyter nbconvert --execute model_development.ipynb` termina sin errores en ≤ 20 min |
| Sáb 31 | Revisión final de Kevin: requisitos 3.1-3.9 cubiertos según el mapa de la sección 1 |
