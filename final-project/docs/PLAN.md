# Plan del proyecto final: riesgo de prórroga en contratos públicos (SECOP II)

> Versión 3.3 (25-oct-2026), con el cierre de la F1: etiqueta aprobada, `docs/LISTA_FUGA.md` v1, límites y reglas de acceso de F2, dígito de la muestra e incidente de Contratos (sección 0, punto 12; límites de tiempo; F1-F3; 4.1-4.8; riesgos).
> La versión 3.2 agregó el calendario, los recortes y los límites de tiempo de Kevin, y lo medido en la F0: entorno, hallazgo de SHAP, fuga de los montos y señal limpia (sección 0, puntos 7 a 11, y lo que de ellos se deriva en F0, J1-J3, G1 y riesgos).
> Leyenda: **[VERIFICADO]** visto en datos/código · **[SUPUESTO]** · **[ESTIMADO]** · **[NO EVALUADO]**.
> Cada tarea va marcada **[MÍNIMO]** (necesaria para cumplir el enunciado) o **[EXTRA]** (fuera del alcance salvo que sobre tiempo).

## Contexto

La exploración (`00_exploracion/`) eligió SECOP II: predecir, al firmar un contrato, la probabilidad de que termine prorrogado, para que veedurías y oficinas de control interno prioricen qué vigilar. El profesor aprobó el tema y pidió **justificar bien cómo los datos del contrato predicen la prórroga y qué ingresa el usuario**. Hay señal, pero más modesta de lo que decía la exploración (sección 0, punto 9): con las features limpias el HGB da 0,751 ROC-AUC / 0,333 PR-AUC en 2024 (F0, base v0, `y_dias`; v1: 0,781 / 0,584) frente a 0,714 / 0,277 de la regla por entidad. Además, la etiqueta subestima los positivos, hay censura y la tasa salta de 12 % a 20 % en 2025. La F1 mostró que ese salto es un efecto de registro y cambió la etiqueta (sección 0, punto 12).

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
7. **SHAP y las categorías nativas del HGB.** [VERIFICADO con datos sintéticos, `scikit-learn` 1.9.0 + `shap` 0.54.1] `TreeExplainer` sobre un `HistGradientBoostingClassifier` con códigos ordinales y sin categorías nativas es exacto (error de aditividad 0,0000). Con `categorical_features=mask` no lanza error, pero las atribuciones son incorrectas (error de aditividad máximo 13,6 log-odds, medio 5,4). La exploración entrenó el HGB con categorías nativas (`exploracion_secop.ipynb`: `categorical_features=mask` sobre un `OrdinalEncoder` limitado a las 200 categorías más frecuentes), así que las cifras de referencia (0,763 / 0,334) dependen de eso. En J2-J3 se optó por códigos ordinales con `categorical_features=None`: la app usa `SHAP TreeExplainer`, que no explica bien las categorías nativas. La decisión se documenta en el notebook con Decisión / Alternativas / Por qué / Evidencia.
8. **Entorno.** Python 3.12 con venv en `final-project/.venv` y versiones fijas en `requirements.txt` (`scikit-learn` 1.9.0, `shap` 0.54.1, `streamlit` 1.66.1, `pytest` 9.1.1). El `requirements.txt` de la raíz dice Python 3.14.3; el del proyecto final usa 3.12 porque `shap` 0.54.1 exige Python ≥ 3.12 y es la versión instalada en la laptop de Kevin. Se omiten `jupyter` y `notebook`: traen `jupyterlab`, que no se instala en Windows sin rutas largas cuando la carpeta del repo es profunda (una ruta de 270 caracteres frente al límite de 260 en la carpeta de Kevin). `nbconvert` + `ipykernel` bastan para `jupyter nbconvert --execute` y para abrir notebooks en VS Code. **Pendiente:** Jose y Geronimo confirman que pueden usar Python 3.12 y Geronimo confirma que Streamlit Community Cloud lo ofrece [SUPUESTO].
9. **Señal limpia medida en la F0 (base v0, `y_dias`).** [VERIFICADO; muestra de la exploración, train 2019-2023 y test 2024 con 23.999 contratos y tasa base 0,123; montaje de la exploración, HGB sin ajustar] Regla por entidad: 0,714 ROC-AUC / 0,277 PR-AUC. HGB sobre las 39 `FEATURES`: 0,751 / 0,333 (2,72 veces la tasa base). HGB A original, con los 6 montos: 0,763 / 0,334. El HGB limpio le gana a la regla por +0,037 de ROC-AUC (IC 95 % con bootstrap por entidad, 200 repeticiones: [+0,011, +0,069]) y +0,056 de PR-AUC ([+0,003, +0,113]).
   - Los 6 montos inflan el resultado: quitarlos de la variante A baja el ROC-AUC de 0,763 a 0,742; sumarlos a las 39 `FEATURES` lo sube de 0,751 a 0,807 y el PR-AUC de 0,333 a 0,429, porque junto con el precio base delatan las adiciones de valor [F0, base v0, `y_dias`; v1: ROC-AUC limpio 0,781, con 6 montos 0,917, con fuga evidente 0,984]. Ese 0,807 / 0,429 se parece mucho a la variante C de la exploración (0,809 / 0,436), cuyas cifras probablemente están infladas por lo mismo y no se deben citar [SUPUESTO: C no se recalculó].
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
   - **ROC-AUC dentro de entidad:** probabilidad de que, entre un contrato prorrogado y uno no prorrogado de la misma entidad, el modelo puntúe más alto al prorrogado (empates 0,5). Se calcula sobre los 501.046 pares de las 486 entidades del test 2024 que tienen ambas clases. La regla por entidad da 0,500 por construcción (puntaje constante dentro de la entidad). HGB limpio: **0,612 [0,524, 0,685]** ponderando por pares y **0,633 [0,609, 0,657]** promediando por entidad (intervalos con bootstrap por entidad). Con solo las entidades de 20 o más contratos (179, el 99 % de los pares) da 0,612 [0,524, 0,686]. El efecto es moderado y ambos intervalos excluyen 0,50, pero con solo las 29 features seguras el intervalo por pares incluye 0,50 (0,592 [0,473, 0,670]; por entidad 0,637 [0,609, 0,661]). Es lo que ve una veeduría que revisa los contratos de una sola entidad, y la regla no puede aportar nada ahí.
   - **Madurez:** filtrar por `eval_madura` casi no cambia las referencias (regla 0,714 → 0,715; HGB 0,751 → 0,752), porque la duración planeada mediana es de 120 días y el 97-99 % de los contratos de 2019-2025 ya venció.
12. **Lo que cambió la F1 (23-25 de octubre).** [VERIFICADO; reportes en `.agents/tasks/f1/`, carpeta local no versionada; resumen de fuga en `docs/LISTA_FUGA.md` v1]
   - **Etiqueta aprobada:** `y = 1` si `dias_adicionados > 0` o existe una modificación `Publicado` con `dias_extendidos > 0` y `fecha_de_aprobacion > fecha_de_firma + 1 día` (4.3). El objetivo se llama **ampliación del plazo**: las suspensiones y los ajustes de fecha cuentan como positivos. En 1.400 contratos al azar (200 por año de firma), `y_dias` da 12,1 % y la etiqueta aprobada 24,4 % [22,2, 26,7] (27,6 % en la base v1).
   - **El salto de 2025 en `y_dias` es un efecto de registro, no una deriva real.** `dias_adicionados` guarda solo los días de la última modificación publicada y vuelve a 0 cuando se publica otra sin días. En los contratos recientes todavía no llegó esa modificación posterior: `y_dias` marca el 80 % de los positivos de 2025 frente al 43 % de 2019-2024. Con la etiqueta aprobada, 2019-2024 queda entre 19,5 % y 26,5 % sin diferencias entre años (p = 0,56) [Actualizado con la v1: sí hay diferencias, de 22,2 % en 2019 a 30,0 % en 2023 y 24,7 % en 2024; el paso 3 tenía ~200 contratos por año y no podía detectarlas. Ver docs/BASE_V1.md.] y 2025 en 30,0 % (p = 0,05 frente a 2019-2024). **Nota para G3:** el ejemplo de alarma pasa a ser este cambio de registro, no un cambio de comportamiento.
   - **Nota para J1-J3:** las cifras de la F0 (0,751 de ROC-AUC, +0,037 sobre la regla, entidades no vistas, top-k y dentro de entidad, puntos 9 y 11) se midieron con `y_dias` sobre la base v0. Las referencias reales se miden con la base v1: ROC-AUC limpio 0,781; con 6 montos 0,917; con fuga evidente 0,984.
   - **`fecha_de_inicio_del_contrato`:** no cambia más en `y = 1` que en `y = 0` (17,3 % frente a 13,4 %, p = 0,37), y casi todo es un desfase de 1 día entre conjuntos. `dias_firma_a_inicio` y `fin_planeado_cruza_anio` quedan como "segura con limitación"; la ablación pasa a J3.
   - **`duraci_n_del_contrato`** pasa a sospechosa (se actualiza en parte) y deja de ser respaldo en la censura (4.5).

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
| 3.7 Métricas | Justificadas en términos de uso | Notebook §3.7: PR-AUC, precisión/recall en top-k, ROC-AUC dentro de entidad (la regla da 0,5 por construcción; sección 0, punto 11), tasa base, regla por entidad | Jose |
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
- **Base v1** (lunes 26 a las 12:00; ver F2): pipeline completo, con la etiqueta aprobada en la F1 (4.3).

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
def roc_auc_intra_entidad(y_true, score, entidad) -> dict                 # por pares y promedio por entidad; la regla por entidad da 0,5
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
| Sáb 24 - Dom 25 | **F1** se extiende al domingo 25 (pasos 4 y 5) + **F2** desde el domingo 25 (termina a más tardar el lunes 26 a las 6:00) + inicio del EDA con v0 | J1 → **J2** | Termina **G1** |
| Lun 26 - Mar 27 | **F3** (base v1 el lunes 26 a las 12:00) → **EDA con v1** (lunes 26 en la tarde y martes 27) | **J2** el lunes 26 con la base v0 (el `Pipeline` no depende de la etiqueta) → **J3** desde el lunes 26 en la tarde, con la v1 | **G2** con el dummy → **G3** (la simulación usa la v1 desde el lunes en la tarde) |
| Mié 28 - Jue 29 | Revisa PRs | **J3** (modelo elegido el jueves) | **G3** + modelo real en la app |
| Vie 30 | **README** | **Integración del notebook** | **Video** |
| Sáb 31 | **Revisión final y entrega** | Correcciones de la revisión | Verifica la app desplegada |
| Lun 2 nov | Ensayo de sustentación (todos) | | |
| Mar 3 nov | **Sustentación** | | |

**Checkpoints:**
- Etiqueta: viernes 23 por la noche (aprobada en el checkpoint del paso 3 de la F1).
- Base v1: lunes 26 a las 12:00 (definitiva; regla de cambio en F2).
- Modelo elegido: jueves 29.
- Notebook integrado: viernes 30 por la noche.

### Límites de tiempo (reglas duras)

| Tarea | Límite | Si se pasa |
|---|---|---|
| Verificación de la etiqueta (F1) | Medio día (4 h desde el 23-oct a las 21:08) | Si no es concluyente: `y = dias_adicionados > 0`, se documenta la limitación y F2 no descarga Modificaciones. **Resultado:** concluyente en 33 min; no se aplicó la salida |
| Descarga F2: tope de planificación | La estimación de F1 no puede pasar de **3 h si F2 corre de día ni de 8 h si corre de noche**, siempre que la descarga sea reanudable. La condición original era terminar antes del domingo 25 a las 8:00; quedó superada por la fila siguiente | Si el exceso viene de Contratos o Procesos, se baja al 5 % (5 de las 100 terminaciones de dos dígitos). **Resultado (F1, paso 4):** central 1 h 18 min, cabe de día y de noche; pesimista 3 h 28 min, cabe solo de noche. Las dos caben en la ventana de ejecución, hasta el lunes 26 a las 6:00. No se aplicó la salida |
| Descarga F2: ejecución | **Continua hasta terminar, sin pausa a las 3 h. Único límite: lunes 26 a las 6:00** (`--hasta 2026-10-26T06:00`). Reanudable: cortar en cualquier momento deja el estado consistente y reanudar no repite unidades hechas | A las 6:00 se detiene limpio (código de salida 4) y se avisa al usuario. Ya no aplica el corte duro del domingo a las 22:00 con `y = y_dias`, porque `y_dias` esconde la mitad de los positivos (12,1 % frente a 24,4 %). Solo se cae a `y_dias` si la descarga de Modificaciones falla del todo, no por retraso |
| Peticiones a datos.gov.co | **1 por segundo, medido de inicio a inicio, en secuencia y sin paralelismo** (reglas de acceso, abajo) | Ante un 403 o un 429 que persiste, se detiene todo (`Throttled`, código de salida 3) y se avisa al usuario |
| Búsqueda de hiperparámetros (J3) | **15 min por modelo** | Se mide un ajuste y se fija `n_iter` para no pasar el tope. kNN y SVM sobre una submuestra pequeña (~20 mil de train) |
| `tests/test_contratos.py` | Solo las 3 verificaciones de 2d | — |

**Reglas de acceso a datos.gov.co** (`robots.txt` de `www.datos.gov.co`, leído en el paso 1 de la F1; decisión del usuario del 23-oct):
- **`Crawl-delay: 1`.** Se respeta en todas las peticiones del proyecto, API incluida: máximo 1 petición por segundo, en secuencia y sin paralelismo. No hay hilos ni dos descargas a la vez.
- **Rate limiter de inicio a inicio (start-to-start).** El segundo se mide entre el inicio de una petición y el inicio de la siguiente, no como espera fija después de cada respuesta. Si una petición tarda más de 1 s, la siguiente sale de inmediato. Lo implementa `LimitadorTasa` en `src/cliente_socrata.py`, compartido por todas las instancias del proceso. La estimación de F2 usa ese modelo: cada petición cuesta max(1 s, latencia).
- **Endpoints prohibidos.** El `robots.txt` prohíbe `/OData.svc/`, `/api/odata/`, `/api/collocate*` y `/views/INLINE/rows.json` (`method=clustered2`), además de varias rutas de `/browse` del catálogo web. El cliente solo acepta una lista blanca: `/resource/<id>.json` (SODA 2.x), `/api/v3/views/<id>/query.json` y `/api/views/<id>(.json)` (metadatos). Cualquier otra ruta falla sin salir a la red.
- **Token:** solo en el header `X-App-Token`; nunca en la URL ni en los logs (4.6).
- **Excepciones aceptadas por el usuario (25-oct):** el monitor de Contratos (1 petición cada 10 min, sin token, `count(*)` de `jbjy-vk9h`) y dos lecturas de la herramienta web del orquestador hacia las 16:35. Los reintentos con `Retry-After` cubren una coincidencia.

### Estimaciones recalculadas [ESTIMADO]

Supuesto: jornadas de ~8 h efectivas por persona y por día, fines de semana incluidos [SUPUESTO].

| Fase | Responsable | Esfuerzo | Ventana | ¿Cabe? |
|---|---|---|---|---|
| F0 | Kevin | 7-8 h | Jue 22 (8 h) | **Justo** |
| F1 | Kevin | 6-7 h (etiqueta ≤ 4 h) | Vie 23 (8 h) | No: se extendió al domingo 25 (pasos 4 y 5); la etiqueta tomó 33 min |
| F2 | Kevin | Código hecho en la F1 (`src/descarga.py`, commit `93f2f0a`) + descarga de 1 h 18 min (central) a 3 h 28 min (pesimista) | Dom 25 - Lun 26 a las 6:00 | Sí, en las dos estimaciones |
| F3 | Kevin | 5-6 h | Lun 26 hasta las 12:00 | **Justo**: arranca al terminar F2 |
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
- **Por qué:** es el único tramo del camino crítico que depende de algo externo (la disponibilidad y la velocidad de la API, el `Crawl-delay` y Modificaciones con 30 M de filas) y concentra ~15 h de Kevin junto con el inicio del EDA. Si la base v1 se corre del lunes 26 a las 12:00 al martes, J3 pierde un día de sus ~3, el EDA con v1 queda en un solo día y el modelo real llega tarde al video.
- **Mitigaciones:**
  - `descarga.py` probado con un mes en la F1 (paso 5) y revisado; commit `93f2f0a`.
  - F2 corre de continuo, reanudable, con el lunes 26 a las 6:00 como único límite. F3 arranca sin esperar aprobación cuando F2 pasa el chequeo de completitud (F3, abajo).
  - Escribir el EDA como funciones sobre v0 para re-ejecutarlo en v1 sin rehacerlo.
  - Contratos quedó truncado el 25-oct: se arman dos bases en dos carriles, con una regla fija para elegir cuál se publica el lunes 26 a las 12:00 (F2, abajo).
  - **Sin corte duro con `y_dias`:** el corte del domingo 25 a las 22:00 ya no aplica. Solo se publica una v1 con `y = y_dias` si la descarga de Modificaciones falla del todo, y la decide el usuario.

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
- **Hecho en la F1** (23-25 oct; cada paso con checkpoint del usuario):
  - **`robots.txt`:** `Crawl-delay: 1`, que se respeta también en la API (reglas de acceso, en los límites de tiempo).
  - **Fecha de inicio:** fuga descartada en lo que el método puede ver; "segura con limitación" (sección 0, punto 12, y `docs/LISTA_FUGA.md`, sección 6). Sin ablación en la F1: pasa a J3.
  - **Etiqueta:** concluyente en 33 min; definición aprobada en 4.3.
  - **Conteo por dígito y mediciones:** 142 peticiones el 24-oct; dígito 4, lotes de 500 IDs y Modificaciones por listas de IDs (4.1 y 4.3). Estimación de F2 en los límites de tiempo.
  - **`src/cliente_socrata.py` y `src/descarga.py`:** probados con un mes (Contratos ya estaba truncado, así que el mes tuvo 6 contratos), con corte y reanudación sobre las 84 unidades de Contratos; revisión de código aprobada; commit `93f2f0a`. Queda la prueba de varios lotes de Procesos y Modificaciones, que cubre la corrida del carril B.
  - **Columnas "por verificar":** solo se revisó `duraci_n_del_contrato` (sospechosa). Las demás siguen por verificar y van a la ablación de J3 (`docs/LISTA_FUGA.md`, sección 5).
  - **`docs/LISTA_FUGA.md` v1** escrita en el cierre de la F1.

**F2. Descarga (dom 25 - lun 26 a las 6:00)**
- Rama: se trabaja en `data/verificacion-etiqueta`, sin cambiar de rama durante F2 y F3. Al llegar a cada commit se proponen la rama y los archivos y se espera la aprobación del usuario. `data/` no se versiona.
- Tareas [MÍNIMO]: `src/descarga.py` (4.6), hecho en la F1 (commit `93f2f0a`) sobre `src/cliente_socrata.py`: timeout de 30 s, 3 intentos y App Token en el header. Comando, desde `final-project/`: `.venv/Scripts/python.exe -m src.descarga --hasta 2026-10-26T06:00`; el mismo comando reanuda.
- Orden, en secuencia: Contratos (84 unidades mes × dígito 4, en páginas de 10.000) → Procesos (~883 lotes de 500 IDs) → Modificaciones (~906 lotes de 500 IDs). Corte único para los tres conjuntos: la hora de arranque, guardada en el manifiesto y aplicada como `fecha_version <= corte` en Modificaciones.
- Listo cuando: el manifiesto no tiene unidades pendientes ni fallidas, el conteo por año × dígito cuadra con el conteo exacto (4.1) y el conteo por año cuadra con el 10 % de la población (±2 %). `completitud.json` lo registra, y `descarga.py` sale con un código distinto de 0 si algo no cuadra o queda una unidad sin hacer.
- **Limitación aceptada por el usuario:** el corte se guarda en hora local sin zona. Si `fecha_version` viene en UTC, el corte efectivo queda hasta 5 h antes de la hora de arranque, lo que es conservador. La corrección va con el modo de Contratos local del carril B y se documenta en `docs/BASE_V1.md`.

**Incidente de Contratos y decisión C (domingo 25 de octubre)**
- **Hecho:** desde el 25-oct a las 00:48 (hora de Colombia), Contratos (`jbjy-vk9h`) tiene solo 1.000 filas en el portal; el 24-oct a las 11:22 tenía 5,6 M. Procesos y Modificaciones están normales. Un monitor sin token lee el conteo cada 10 minutos.
- **Decisión C del usuario: dos bases en dos carriles.**
  - **Carril B (esa noche):** base v1 sobre la muestra de la exploración (foto del 14-oct: 167.997 contratos de 2019-2025, 2.000 por mes). Contratos se lee de los archivos locales y solo se descargan Procesos y Modificaciones, con corte el 2026-10-14 a las 22:53 (hora de Colombia). Luego F3 sobre B.
  - **Carril A (condicional):** F2 normal (dígito 4, 452.515 contratos) si Contratos vuelve a un conteo normal antes del lunes 26 a las 2:30. Se lanza sin esperar al usuario, con `--hasta 2026-10-26T06:00`, y luego F3 sobre A.
  - Nunca corren dos descargas a la vez. Si A arranca con una descarga de B en curso, gana A: B se pausa con corte limpio y se reanuda cuando A termina. F3 de B sí puede solaparse con la descarga de A, porque F3 no usa la red.
- **Regla de las 12:00:** la base que se publica el lunes 26 a las 12:00 es la definitiva. Si A terminó y el usuario aprobó su reporte de calidad antes de esa hora, se publica A; si no, B. No se cambia después de las 12:00.
- **Desviación de B respecto a lo aprobado:** lo aprobado era el 10 % por dígito 4, proporcional por año. B es un muestreo estratificado por mes con asignación igual (2.000 por mes), así que no es autoponderada y lleva pesos por mes.
- **Qué base quedó y por qué** se documenta al final de la corrida B/A (`docs/BASE_V1.md`), no en el cierre de la F1.

**F3. Base v1 y calidad (lun 26, hasta las 12:00)**
- Rama: se trabaja en `data/verificacion-etiqueta`, como F2. La rama del commit (`data/base-contratos-v1` en el plan original) se propone al llegar a él.
- Arranca sin esperar aprobación cuando F2 termina y pasa el chequeo de completitud. Se detiene en el reporte de calidad y antes de cada commit, y también si algo falla o un chequeo no pasa.
- Tareas [MÍNIMO]:
  - `src/etiqueta.py` con la definición aprobada (4.3). Deduplica las filas exactas de Modificaciones (10,4 % en la muestra del paso 3 y 15,4 % en la del paso 4), filtra `fecha_version <= corte` y guarda `y`, `y_dias` e `y_modif`.
  - Censura (4.5), `src/calidad.py` (4.7) y la base v1 con su `meta.json` y sha256.
  - Los cambios de código de `docs/LISTA_FUGA.md`, sección 8: `COLUMNAS_FUGA`, estado de las dos features de la fecha de inicio y lista PII de `src/config.py`.
- [EXTRA] Detector de columnas mutables.
- Listo cuando: la calidad pasa y la base v1 está compartida con su sha256 el lunes 26 a las 12:00.

**F4. EDA (sáb 24 - mar 27)**
- Rama: `feat/eda`, en el borrador `notebooks/03_eda.ipynb`.
- Tareas [MÍNIMO]:
  - **H1-H7 con evidencia** (tasa de prórroga por grupo con intervalo, gráfica e interpretación): responde al profesor cómo los datos de la firma predicen la prórroga.
  - Etiqueta `y` frente a `y_dias` por año.
  - Censura y duración; deriva; evidencia de fuga (contraste 0,95 / 0,84; F0, base v0, `y_dias`; en la v1: ROC-AUC limpio 0,781, con 6 montos 0,917, con fuga evidente 0,984).
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
  - Implementar `roc_auc_intra_entidad(y_true, score, entidad)`, que devuelve la versión ponderada por pares y el promedio por entidad (solo entidades con ambas clases), con IC por bootstrap de entidades. Referencia con el HGB limpio en 2024: 0,612 [0,524, 0,685] por pares y 0,633 [0,609, 0,657] por entidad; la regla da 0,5.
- Listo cuando: con la base v0, la regla por entidad da ≈ 0,714 / 0,277 en 2024 (±0,005, cifras de la exploración) y el HGB sobre `FEATURES` da ≈ 0,751 / 0,333 (F0, base v0, `y_dias`; referencia de la sección 0, punto 9). Ese montaje usa categorías nativas sobre un `OrdinalEncoder` de las 200 categorías más frecuentes, `random_state=42` y el resto por defecto (sección 0, punto 7). El HGB A original (0,763 / 0,334) incluía los 6 montos y ya no es la referencia. En la v1 (base v1_A, `y`): ROC-AUC limpio 0,781; con 6 montos 0,917; con fuga evidente 0,984.

**J2. Preprocesamiento (sáb 24 - lun 26)**
- Rama: `feat/preprocesamiento`.
- Tareas [MÍNIMO]: `ColumnTransformer` por familia de modelo (categorías nativas para HGB; one-hot + escalado para kNN y SVM; `TargetEncoder` para la entidad), faltantes y atípicos, todo justificado. Para el HGB deja listas las dos variantes (con y sin categorías nativas), porque J3 las compara por su efecto en SHAP (sección 0, punto 7).
- Listo cuando: todo vive en el `Pipeline` y se ajusta solo con train. Ese `Pipeline` es el que Geronimo usa en G3.

**J3. Modelado y selección (desde el lun 26 en la tarde, con la v1, hasta el jue 29)**
- Rama: `feat/modelado`.
- Tareas [MÍNIMO]:
  - kNN, árbol, RF, HGB y SVM, con búsqueda sobre `folds_temporales` y **tope de 15 min por modelo**; kNN y SVM sobre submuestra.
  - Early stopping desactivado o temporal.
  - Variante C sin `visualizaciones_del`.
  - Test 2024 una sola vez; deriva 2025 (i) y, si da el tiempo, (ii).
  - **SHAP global enlazado a H1-H7**; sensibilidad `y` frente a `y_dias`.
  - **Ablación explícita: HGB final sin `dias_firma_a_inicio` ni `fin_planeado_cruza_anio`, contra la regla, global y dentro de entidad. Si la ventaja cae de forma importante, se documenta como limitación en 3.6.** (Decisión del checkpoint del paso 2 de la F1; `docs/LISTA_FUGA.md`, sección 6.)
  - Las referencias de la F0 se midieron con `y_dias` sobre la base v0; las de J3 se miden con la base v1 (sección 0, punto 12).
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
  - Simulación con el `Pipeline` de J2 + HGB por defecto entrenado en 2019-2023, recorriendo 2024-2025 mes a mes (o por trimestre si se aprieta), para mostrar dónde salta la alarma. Usa la base v1 desde el lunes 26 en la tarde.
  - **Ejemplo de alarma:** el salto de 2025 en `y_dias` es un cambio de registro (`dias_adicionados` guarda solo la última modificación publicada), no un cambio de comportamiento (sección 0, punto 12). La simulación lo presenta así.
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
- **Tamaño:** con el dígito 4, 452.515 contratos 2019-2025 (train 261.842, test 2024 88.548, test 2025 102.125) [VERIFICADO, conteo del 24-oct].
- **Costo:** descarga de 1 h 18 min (central) a 3 h 28 min (pesimista) [ESTIMADO en la F1, paso 4]; pico de RAM < 2-3 GB procesando por meses; base v1 de ~150-300 MB en memoria [ESTIMADO].
- **Dígito: el 4, no el sorteado** (checkpoint del paso 4 de la F1). El dígito sorteado con `default_rng(42)` fue el 0. F2 usa el 4 por estas razones, en este orden:
  1. El criterio es la proporcionalidad por año. El 0 se desvía -6,8 % en 2019 y no cumple el ±2 % relativo en 2019-2022.
  2. El 4 es el único, junto con el 5, que queda dentro de ±2 % en todos los años (desvío máximo de 1,39 % el 4 y 0,21 % el 5). Entre los dos se prefiere el 4 por su menor desviación en la tasa de `y_dias`: 0,27 pp como máximo, frente a 0,94 pp del 5 en 2019.
  3. La tasa de `y_dias` se usó solo como control, no como criterio de selección: solo desempató entre el 4 y el 5, que ya cumplían el criterio de tamaño.

**Contratos por año de firma y último dígito** [VERIFICADO; 10 consultas `count(*)` del 2026-10-24 a las 11:22]. La suma de los 10 dígitos es igual al total sin filtro en todos los años. 2019-2024 coincide con `meta/poblacion_contratos_por_anio.csv` (14-oct) y 2025 tiene 1 contrato más. La columna del dígito 4 es el conteo exacto contra el que F2 hace el chequeo de completitud.

| Año | 0 | 1 | 2 | 3 | **4** | 5 | 6 | 7 | 8 | 9 | Población | 10 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019 | 13.240 | 15.751 | 15.056 | 14.622 | **14.407** | 14.239 | 13.881 | 13.816 | 13.649 | 13.429 | 142.090 | 14.209 |
| 2020 | 34.097 | 37.996 | 36.952 | 36.378 | **36.107** | 35.670 | 35.246 | 35.159 | 34.649 | 34.369 | 356.623 | 35.662 |
| 2021 | 54.306 | 58.544 | 57.421 | 56.857 | **56.469** | 56.127 | 55.707 | 55.386 | 55.025 | 54.599 | 560.441 | 56.044 |
| 2022 | 69.303 | 73.292 | 72.022 | 71.406 | **71.095** | 70.901 | 70.406 | 70.015 | 69.853 | 69.472 | 707.765 | 70.777 |
| 2023 | 81.943 | 86.446 | 84.923 | 84.074 | **83.764** | 83.621 | 83.112 | 82.797 | 82.551 | 82.231 | 835.462 | 83.546 |
| 2024 | 86.377 | 90.788 | 89.465 | 88.924 | **88.548** | 88.243 | 87.875 | 87.830 | 87.196 | 86.732 | 881.978 | 88.198 |
| 2025 | 99.935 | 104.669 | 103.018 | 102.412 | **102.125** | 101.663 | 101.378 | 100.918 | 100.638 | 100.319 | 1.017.075 | 101.708 |
| **2019-2025** | 439.201 | 467.486 | 458.857 | 454.673 | **452.515** | 450.464 | 447.605 | 445.921 | 443.561 | 441.151 | 4.501.434 | 450.143 |

**Desvío del tamaño de cada dígito frente al 10 % exacto** (%, relativo):

| Año | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2019 | -6,82 | +10,85 | +5,96 | +2,91 | +1,39 | +0,21 | -2,31 | -2,77 | -3,94 | -5,49 |
| 2020 | -4,39 | +6,54 | +3,62 | +2,01 | +1,25 | +0,02 | -1,17 | -1,41 | -2,84 | -3,63 |
| 2021 | -3,10 | +4,46 | +2,46 | +1,45 | +0,76 | +0,15 | -0,60 | -1,17 | -1,82 | -2,58 |
| 2022 | -2,08 | +3,55 | +1,76 | +0,89 | +0,45 | +0,18 | -0,52 | -1,08 | -1,30 | -1,84 |
| 2023 | -1,92 | +3,47 | +1,65 | +0,63 | +0,26 | +0,09 | -0,52 | -0,90 | -1,19 | -1,57 |
| 2024 | -2,06 | +2,94 | +1,44 | +0,82 | +0,40 | +0,05 | -0,37 | -0,42 | -1,14 | -1,66 |
| 2025 | -1,74 | +2,91 | +1,29 | +0,69 | +0,41 | -0,04 | -0,32 | -0,78 | -1,05 | -1,37 |
| **Máx. absoluto** | 6,82 | 10,85 | 5,96 | 2,91 | **1,39** | **0,21** | 2,31 | 2,77 | 3,94 | 5,49 |
- **Control de RAM:**
  - Procesar mes a mes y solo con las columnas necesarias.
  - Tipos explícitos (category, float32, Int16, `string[pyarrow]`).
  - Nada de `dtype=str` para la tabla entera.
  - Registrar el pico con `psutil`.
- **Otras opciones:** el 20 % es [EXTRA]; el 5 % es la salida si la estimación de F2 pasa del tope (3 h de día, 8 h de noche; no hizo falta); la población completa se descarta.

### 4.2 Columnas a descargar (verificadas contra `meta/*.json`) [VERIFICADO]

**Contratos (`jbjy-vk9h`):**
- Llaves y momento: `id_contrato`, `proceso_de_compra`, `referencia_del_contrato`, `fecha_de_firma`.
- Entidad: `nombre_entidad`, `nit_entidad`, `departamento`, `ciudad`, `orden`, `sector`, `rama`, `entidad_centralizada`.
- Diseño: `tipo_de_contrato`, `modalidad_de_contratacion`, `justificacion_modalidad_de`, `codigo_de_categoria_principal`, `objeto_del_contrato`, `es_grupo`, `es_pyme`, `tipodocproveedor`, `habilita_pago_adelantado`, `valor_de_pago_adelantado`, `origen_de_los_recursos`, `destino_gasto`, `espostconflicto`, `documentos_tipo`, `el_contrato_puede_ser_prorrogado` (por verificar; excluida de `FEATURES`, sección 0, punto 10).
- Fuentes de recursos (fuga confirmada; solo en `data/raw/` para el EDA de fuga, nunca en la base): `presupuesto_general_de_la_nacion_pgn`, `sistema_general_de_participaciones`, `sistema_general_de_regal_as`, `recursos_propios_alcald_as_gobernaciones_y_resguardos_ind_genas_`, `recursos_de_credito`, `recursos_propios`.
- Fechas: `fecha_de_inicio_del_contrato` (segura con limitación) y `duraci_n_del_contrato` (sospechosa: solo en `data/raw/`, nunca en la base), según `docs/LISTA_FUGA.md` v1.
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
- Verificación (solo en la F1, ya hecha): `id_contrato`, `identificador`, `identificador_modificacion`, `numero_version`, `version_anterior_del_contrato`, `estado_modificacion`, `dias_extendidos`, `fecha_creacion`, `fecha_de_aprobacion`, `fecha_version`, `fecha_inicio_contrato`, `fecha_fin_contrato`, `proposito_modificacion`, `descripcion`, `valor_modificacion`.
- Etiqueta (F2), con el filtro de servidor `estado_modificacion = 'Publicado' AND dias_extendidos > 0 AND fecha_version <= corte`: `id_contrato`, `identificador_modificacion`, `version_anterior_del_contrato`, `fecha_version`, `estado_modificacion`, `dias_extendidos`, `fecha_de_aprobacion`, `fecha_fin_contrato`. `identificador_modificacion`, `version_anterior_del_contrato` y `fecha_version` sirven para deduplicar y ordenar.
- **No se descargan:** `autor_version` y `sustituto_version` (PII ampliada; `sustituto_version` puede ser un ID de usuario), `proposito_modificacion` (texto libre que a veces trae nombres de personas; pasa a la lista PII) y `descripcion` (es el objeto del contrato). `src/descarga.py` lo impide con un `assert`.

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

**Resultado de la F1 (paso 3, 23-oct de 21:08 a 21:41): concluyente; definición aprobada en el checkpoint.**

> `y = 1` si `dias_adicionados > 0` **o** existe una modificación con estado `Publicado`, `dias_extendidos > 0` y `fecha_de_aprobacion > fecha_de_firma + 1 día`.

- **"Efectiva" = `Publicado`** (P2). El fin vigente en Contratos es el de la última modificación publicada en 1.520 de 1.521 contratos, y ninguna modificación no publicada se refleja en Contratos.
- **Margen de 1 día** al comparar `fecha_de_aprobacion` (Modificaciones) con `fecha_de_firma` (Contratos), por el desfase de zona horaria del paso 2 (P3). Solo 3 de 1.546 extensiones caen dentro del margen, todas en un contrato.
- **Nombre del objetivo: "ampliación del plazo"** (P4): el contrato terminó después de lo pactado al firmar. Las suspensiones y los ajustes de fecha cuentan como positivos y son cerca del 2,4 % de los contratos positivos. El 69 % de las extensiones son prórrogas o adiciones y el 25 % tiene texto genérico, así que la proporción real de suspensiones no se conoce.
- **Se mantiene el "o `dias_adicionados > 0`"** (P5): frente a usar solo Modificaciones cambia un contrato. Todo `y_dias = 1` tiene su modificación (396 de 396).
- **`dias_extendidos`** es incremental en 2021-2025 y acumulado en 2020-2021 y en 2026 (P1). Para la etiqueta binaria no importa. Para contar días se usa el cambio de `fecha_fin_contrato`, no `sum` ni `max`.
- **Tasas** (1.400 contratos al azar, 200 por año de firma): `y_dias` 12,1 % y la etiqueta aprobada 24,4 %. En 2024 pasa de 9,0 % a 21,5 % [16,4, 27,7]; corrigiendo por la captura de `y_dias`, la tasa poblacional de 2019-2024 sería ~26 % [ESTIMADO]. La hipótesis del EDA queda apoyada con un matiz (sección 0, punto 12).
- **F2 descarga Modificaciones** con el filtro de servidor de 4.2, que da los mismos contratos positivos que el análisis por modificación (788 de 788).

**Costo de la etiqueta combinada:** ~906 lotes de 500 IDs, en secuencia a 1 petición por segundo, de 55 min (central) a 135 min (pesimista) [ESTIMADO en la F1, paso 4]. La estimación anterior (~3.000 lotes de 150 IDs, 8-35 min con 3 hilos) suponía un paralelismo que el `Crawl-delay` no permite. Pedir Modificaciones por rango de fechas resultó más lento y poco fiable: el 45 % de las consultas mensuales pasó de 30 s.

**Volumen de Modificaciones (filas por contrato).** Las dos muestras de la F1 incluyen los duplicados exactos, así que el 0,57 del paso 3 y el 0,69 del paso 4 no se concilian restando el 15,4 % de duplicados:

| Base | Paso 3 (1.400 contratos, 200 por año) | Paso 4 (7.000 contratos del dígito 0 de la base v0) |
|---|---|---|
| Filas por contrato, con duplicados | 0,569 | 0,692 |
| Filas por contrato, sin duplicados exactos | 0,511 | 0,585 |
| Duplicados exactos (columnas mínimas) | 10,2 % | 15,4 % |
| Reponderado por la población, con duplicados | 0,604 | 0,628 |
| Reponderado por la población, sin duplicados | 0,533 | 0,529 |

- Ninguna de las dos muestras sigue a la población: el paso 3 pesa los 7 años por igual, y el paso 4 viene de la base v0, donde 2025 pesa 12,7 % (22,6 % en la población). Reponderadas, la diferencia baja a 0,024 con duplicados y a -0,004 sin ellos. El resto es variación muestral: el IC 95 % de la diferencia es [-0,091, +0,139] con duplicados.
- **Volumen para el dígito 4:** ~284.000 filas con duplicados [IC 95 % ~262.000-307.000] y ~239.000 sin ellos [ESTIMADO]. La cifra de ~313.000 (0,692 sin reponderar) queda como cota alta. El número de peticiones no cambia.

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
- `eval_fin_planeado` = inicio (o la firma, si falta) + duración planeada. La duración sale solo del proceso, sin respaldo: la F1 encontró que `duraci_n_del_contrato` se actualiza en parte (sospechosa en `docs/LISTA_FUGA.md`). Si falta la duración del proceso (0,26 % de los contratos en la v0), `eval_madura` queda en `False`.
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
  - Timeout de 30 s y 3 intentos con espera creciente. En `descarga.py`, la espera es exponencial y respeta `Retry-After`; 403 o un 429 que persiste detienen todo (`Throttled`).
  - En secuencia, sin hilos: 1 petición por segundo de inicio a inicio, por el `Crawl-delay: 1` (reglas de acceso, en los límites de tiempo de la sección 3).
  - Lotes de 500 IDs para Procesos y Modificaciones; si la URL preparada pasa de 15.000 caracteres, el lote se parte en dos (en la F1 el servidor respondió 200 con 15.592 caracteres y 414 con 17.299).
  - Paginación estable con `$order=:id` (en la F1, la misma consulta en una página o en cuatro dio las mismas filas en el mismo orden).
  - Log ASCII con lote i/N, filas, tiempo transcurrido, ETA y RAM.
- **App Token:**
  - Variable `SOCRATA_APP_TOKEN`. En local se lee de `final-project/.env` con un lector simple de líneas (sin `python-dotenv`); en la app, de `st.secrets`.
  - Se envía solo en el header `X-App-Token` [VERIFICADO en la F1 con la documentación de Socrata; nunca va en la URL ni en los logs].
  - El token no se comparte por chat ni se sube al repo (que es público): Kevin crea el `.env` y el código solo lee la variable.

### 4.7 Chequeos de calidad (`src/calidad.py`, fallan con error)

1. `id_contrato` único; el join no duplica; ninguna columna PII; `FEATURES ∩ COLUMNAS_FUGA = ∅`.
2. Tamaño por año = 10 % de la población (±2 %), con `meta/poblacion_contratos_por_anio.csv`, y completitud contra el conteo exacto por año × dígito (tabla de 4.1). El conteo exacto detecta lotes que faltan; el 10 % solo mide representatividad.
3. Tasa de `y_dias` por año en la muestra frente a la población (±1,5 pp).
4. Rangos de fechas (años 0025 y 8201 → NaT), montos ≥ 0, duración entre 1 y 3.650 días.
5. ≥ 95 % de los contratos con proceso; faltantes por feature; dtypes iguales al esquema.
6. Fracción madura por año; `y` frente a `y_dias` por año.
7. Ninguna feature con ROC-AUC univariado > 0,85 sin revisión.

### 4.8 `docs/LISTA_FUGA.md`

**La lista vigente es `docs/LISTA_FUGA.md` (v1, cierre de la F1).** Esta tabla es la versión preliminar del plan, con las filas que cambiaron en la F1 ya actualizadas.

Columnas de la tabla: **Columna o derivado | Conjunto | Por qué es fuga | Evidencia | Estado (confirmada / sospechosa / etiqueta / por verificar / descartada, segura con limitación) | Uso permitido (ninguno / solo etiqueta / solo censura / solo EDA / solo ablación / feature)**. `COLUMNAS_FUGA` es su espejo ejecutable; los cambios de código que faltan están en `docs/LISTA_FUGA.md`, sección 8.

| Columna | Estado | Evidencia o motivo | Uso |
|---|---|---|---|
| `estado_contrato` | Confirmada | AUC 0,94 | Solo EDA |
| `fecha_de_fin_del_contrato` y derivados | Confirmada | Razón de duración 1,44 frente a 1,00; AUC 0,71. F1: es el fin de la última modificación publicada (o 1 día antes) en 1.520 de 1.521 contratos | Solo EDA |
| `valor_del_contrato` | Confirmada | Razón de valor 1,18 frente a 1,00 | Solo EDA |
| Pagos, facturado, amortizado, pendientes, saldos (8 columnas) | Confirmada | Ejecución posterior (pendientes: AUC 0,63) | Ninguno |
| `ultima_actualizacion` | Confirmada | AUC 0,74 | Ninguno |
| Liquidación y reversión (6 columnas) | Confirmada | Posteriores al contrato | Ninguno |
| `fecha_de_notificaci_n_de_prorrogaci_n` | Confirmada | Registra la propia prórroga | Ninguno |
| `dias_adicionados`; dataset Modificaciones | Etiqueta | Definen `y`. F1: `dias_adicionados` copia los días de la última modificación publicada y vuelve a 0 con modificaciones posteriores sin días (sesgo de captura) | Solo etiqueta |
| Procesos: visualizaciones, estados, fase, última publicación | Confirmada (reglas) | Estados actuales o acumulados | Ninguno |
| Ejecución y Adiciones (datasets) | Confirmada | Posteriores a la firma | Ninguno |
| 6 montos de fuentes de recursos | Confirmada | Su suma es igual a `valor_del_contrato` en el 80,6 %; cociente sobre el precio base de 1,26 con adición en el valor frente a 1,00 | Solo EDA |
| `fecha_de_inicio_del_contrato` (y `dias_firma_a_inicio`, `fin_planeado_cruza_anio`) | Descartada, segura con limitación | F1: cambia en el 17,3 % de los `y = 1` frente al 13,4 % de los `y = 0` (p = 0,37), casi todo por un desfase de 1 día. No ve cambios anteriores a la primera modificación | Feature; ablación en J3 |
| `duraci_n_del_contrato` | Sospechosa | F1: es igual al plazo ya extendido en el 24 % de los `y_dias = 1` | Ninguno (no es respaldo en la censura) |
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
| **Fin de semana de Kevin (F2 + F3): camino crítico con dependencia externa** | `descarga.py` probado en la F1 (commit `93f2f0a`); F2 continua y reanudable hasta el lunes 26 a las 6:00; F3 encadenada; base v1 definitiva el lunes 26 a las 12:00. Ya no hay corte duro con `y = y_dias`: solo si la descarga de Modificaciones falla del todo |
| **Contratos (`jbjy-vk9h`) truncado a 1.000 filas desde el 25-oct a las 00:48** | Decisión C: carril B con la foto local del 14-oct y carril A si Contratos se recupera antes del lunes 26 a las 2:30; regla de las 12:00 para elegir la base (F2) |
| **Integración en un solo día (vie 30)** | Borradores en formato final el jueves por la noche; esqueleto del notebook el jueves; resultados de la búsqueda guardados; ejecución ≤ 20 min; revisión de Kevin el viernes por la noche |
| **SHAP da atribuciones incorrectas sin avisar:** `TreeExplainer` con las categorías nativas del HGB (error de aditividad hasta 13,6 log-odds) [VERIFICADO con datos sintéticos] | Prueba de aditividad obligatoria en G1 y en el modelo final; decisión (a) `shap.Explainer` genérico o (b) sin categorías nativas, medida en J3; par fijado en la F0 (`scikit-learn` 1.9.0 + `shap` 0.54.1); los metadatos guardan las versiones |
| Límite de uso de la API | App Token; `Throttled`; caché reanudable; en secuencia, 1 petición por segundo de inicio a inicio |
| `Crawl-delay: 1` aplica a la API | Resuelto en la F1: se respeta (reglas de acceso, sección 3). La estimación de F2 ya lo incluye y cabe, así que no se bajó al 5 % |
| Etiqueta no concluyente | Resuelto en la F1: concluyente en 33 min (4.3) |
| RAM en laptops | Procesar por mes, tipar, submuestra para kNN y SVM, `psutil` |
| Censura | Fin planeado, solo contratos maduros, métricas por duración |
| Fuga sutil | Columnas "por verificar", alarma de AUC univariado, variante C recalculada |
| **La ventaja del modelo limpio sobre la regla por entidad es modesta** (ROC-AUC +0,037 [+0,011, +0,069]; PR-AUC +0,056 [+0,003, +0,113]) y depende de 3 features que usan `fecha_de_inicio_del_contrato`. La F1 la dejó como "segura con limitación": no ve cambios anteriores a la primera modificación. | Ablación explícita de las dos features en J3, global y dentro de entidad; medir el desempeño en entidades no vistas y la precisión en los primeros *k*; etiqueta mejorada, muestra del 10 % y ajuste de hiperparámetros en J3. Si al final no gana con claridad, se reporta con honestidad y se evalúa el valor del modelo en entidades no vistas y su combinación con la regla. Medido en la F0 (sección 0, punto 11): ventaja clara en entidades no vistas (ROC-AUC +0,208 [+0,084, +0,300]) y, en el top-k, solo en el 1 % más alto frente a la regla suavizada. |
| Entidades no vistas | `TargetEncoder` suavizado; IC por entidad; aviso en la app |
| Deriva de 2025 | Test aparte; riesgo en percentil; G3 la usa como alarma. La F1 mostró que el salto de `y_dias` en 2025 es un efecto de registro (sección 0, punto 12) |
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
| 11 | PII ampliada: `codigo_proveedor`, `nombre_del_adjudicador`, `nombre_del_proveedor`, `nit_del_proveedor_adjudicado`, `codigoproveedor`, `autor_version`. **F1 (23-oct):** se agregan `sustituto_version` (puede ser un ID de usuario; no se confirmó qué es) y `proposito_modificacion` (texto libre que a veces trae nombres de personas). Falta reflejarlo en `src/config.py` (F3) |

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
| Lun 26 (12:00) | `calidad.py` sin errores con la base v1 |
| Jue 29 | El artefacto v1 pasa `cargar_modelo` y la app desplegada lo usa |
| Vie 30 | `jupyter nbconvert --execute model_development.ipynb` termina sin errores en ≤ 20 min |
| Sáb 31 | Revisión final de Kevin: requisitos 3.1-3.9 cubiertos según el mapa de la sección 1 |
