# EDA para preprocesamiento (J2)

## Actualización con la base v1 (26-oct)

Base v1_A: 452.494 contratos 2019-2025, 10 % por dígito 4 del id_contrato,
autoponderada. Reemplaza las cifras de la v0, cuya muestra (2.000 contratos
por mes) distorsionaba la composición por año.

Etiqueta. Tasa de y de 27,6 %: 2019 22,2 %, 2020 29,3 %, 2021 27,8 %,
2022 29,4 %, 2023 30,0 %, 2024 24,7 %, 2025 27,1 %. Train (2019-2023)
tiene más positivos que test 2024: afecta PR-AUC y umbrales, no ROC-AUC.

Deriva real 2019 → 2025 (la de la v0 era un artefacto de la muestra):
contratación directa 59,6 % → 75,6 %; prestación de servicios
79,6 % → 85,6 %; proveedor con cédula 75,6 % → 84,9 %; segmento UNSPSC 80
62,6 % → 72,3 %. Medianas: p_dur_dias 180 → 120 días;
dias_publicacion_a_firma 4 → 1.

Acciones para J2:
- p_dur_dias: winsorizar antes de kNN o SVM. Máximo de 4.078 millones de
  días; 5,5 % tiene duración 0 (su tasa, 24,7 %, es parecida al promedio,
  así que un indicador de duración cero es opcional).
- objeto_del_contrato con el texto No definido (560 contratos, tasa 9 %):
  tratarlo como faltante. Si se hace en features.py y no en el Pipeline,
  avisar a Kevin: cambia la app y hay que repetir la paridad.
- len_objeto: los objetos de 50 caracteres o menos tienen 58 % de tasa,
  pero no es fuga: son cargos de personal de hospitales (auxiliar de
  enfermería, médico general) y funciona como indicador indirecto de la
  entidad. Revisar su peso en la evaluación con entidades no vistas.
- Competencia (p_respuestas_al_procedimiento, p_proveedores_invitados):
  más del 90 % vale 0 por la contratación directa. La señal está en la
  cola: 10 o más respuestas, 49 %.
- Relaciones no monótonas (dias_publicacion_a_firma más de 120 días: 15 %
  frente a 40 % entre 31 y 60; duración relativa con pico en 1,5-2): para
  modelos lineales o kNN conviene discretizar; los árboles las capturan
  solos.
- Categorías con mucha señal y pocos casos (licitación de obra pública
  79 %, interventoría 72 %, concurso de méritos 66 %): agrupar las raras
  sin mezclar estas.

Señales más fuertes (para J3):
- Historial de la entidad (H6, sobre 2024): 41,6 % en el tercil alto frente
  a 14,3 % en el bajo y 11,5 % en entidades sin historia. Respalda
  TargetEncoder para nit_entidad.
- Mes de firma (H4): enero-febrero 34 %, noviembre 16 %; último trimestre
  19 % frente a 30 %.
- Restricción para la app: si el modelo es HistGradientBoosting, sin
  categorías nativas (SHAP no las explica bien). Usar TargetEncoder u
  one-hot dentro del Pipeline.

## 0. Lo que necesita J2

- `construir_features` entrega **39 features**: 18 numéricas y 21 categóricas. No hay que modificar `src/features.py`; el `Pipeline` consume `X`.
- Filtrar **`eval_madura = True`** para entrenar y evaluar. `eval_*` nunca entra como feature.
- **Split temporal:** train 2019–2023, test 2024 y 2025 como test de deriva. Nada se ajusta con 2024/2025.
- Todo lo que aprende estadísticas (imputación, escalado, one-hot, target encoding) va **dentro del `Pipeline`** y se ajusta solo con train. El `TargetEncoder` de `nit_entidad` se ajusta **solo con el fold de entrenamiento**; las entidades no vistas reciben la media global.
- Elimina `p_proveedores_que_manifestaron`: es constante 0 (349 nulos).

## 1. Faltantes

| Bloque | n | % | Acción |
|---|---:|---:|---|
| Sin fila de Procesos (todas las `p_*`) | 349 | 0,08 | Imputar mediana/moda + `sin_proceso`. No imputar 0: 0 significa "sin respuesta". |
| `fecha_de_inicio_del_contrato` | 4.059 | 0,90 | `dias_firma_a_inicio` queda NaN: imputar mediana + `sin_inicio`. |
| Duración planeada (`p_dur_dias`) | 799 | 0,18 | NaN: imputar mediana + `sin_duracion`. |
| `p_fecha_de_publicacion_del` | 380 | 0,08 | `dias_publicacion_a_firma` NaN: imputar mediana + `sin_publicacion`. |
| `unspsc_familia` / `unspsc_segmento` | 3.806 | 0,84 | Categoría `SIN_UNSPSC` (son faltantes en la v1); no borrar filas. |
| `p_unidad_de_duracion` | 349 | 0,08 | Categoría `SIN_DATO`. |
| `y_modif` | 0 | 0 | No es feature. |

**No borrar filas.** Los bloques son pequeños y borrarlos sesga la muestra. Los indicadores de faltante son parte de la señal: un contrato sin proceso no es lo mismo que uno con proceso y cero respuestas.

## 2. Valores imposibles y atípicos

| Variable | Hallazgo | Acción |
|---|---|---|
| `p_dur_dias` | 24.566 (5,43 %) ≤ 0; 125 (0,03 %) > 3.650 días; máximo 4.078,9 millones | `<= 0` → NaN; recortar a `[1, 3.650]`; `sin_duracion`; `log1p` solo para kNN/SVM. |
| `p_precio_base` | 10 negativos, 3.497 ceros, máximo 2,5 billones | `log_precio_base` ya viene en `log1p` con `clip(lower=0)`: los negativos quedan en 0 y no se separan de los ceros reales. Crear `precio_no_positivo = log_precio_base == 0`; no volver a aplicar log. |
| `valor_de_pago_adelantado` | 99,89 % en cero; máximo 25.178 millones; 91 "No" y 54 "No Definido" con valor > 0; 1 "Si" con valor 0 | Usar `log_pago_adelantado` (ya en `log1p`) + `tiene_pago_adelantado`; conservar `habilita_pago_adelantado`. No corregir a mano. |
| `p_numero_de_lotes` | 91,60 % en cero; máximo 216 | El cero es "sin lotes", no faltante. `tiene_lotes` + `log1p` para distancia. |
| Conteos de competencia | 90,4–99,8 % en cero | El cero es "sin respuesta/invitación", no faltante. No imputar; `log1p` para distancia; flags de cero si se quiere. |
| `p_proveedores_que_manifestaron` | Constante 0 | Eliminar en todos los modelos. |
| `len_objeto` | Máximo 500 caracteres; 0 ceros | No es un problema de outliers; escalar para distancia. |
| `dias_firma_a_inicio`, `dias_publicacion_a_firma` | Ya vienen recortadas (`[-365, 365]` y `[-30, 1000]`) | No volver a recortar; crear `inicio_retroactivo` y `publicacion_tardia` si el modelo las aprovecha. |

No eliminar outliers: son pocos y pueden ser señal. Para HGB, usar el recorte suave de duración y dejar el resto crudo. Para kNN/SVM, `log1p` + `RobustScaler`.

## 3. Cardinalidad y codificación

| Columna | Categorías | Acción |
|---|---:|---|
| `nit_entidad` | 3.034 | **TargetEncoder** siempre dentro del `Pipeline`, ajustado solo con el fold de entrenamiento (`cv=5`, `smooth="auto"`, `target_type="binary"`); entidades no vistas → media global. No one-hot. Es la señal de H6. |
| `ciudad` | 735 | Agrupar <1 % en `OTROS` + one-hot; para HGB, códigos ordinales. `No Definido` (21,37 %) es categoría, no faltante. |
| `unspsc_familia` | 376 | Igual que `ciudad`; `SIN_UNSPSC` como categoría. |
| `unspsc_segmento` | 56 | Igual; 50 categorías <1 %. |
| `justificacion_modalidad_de` | 41 | One-hot con agrupación <1 %; HGB ordinal. |
| `departamento` | 34 | Igual; `No Definido` (1,79 %) es categoría. |
| `sector` | 26 | Igual. |
| `tipo_de_contrato` | 24 | Igual; 19 categorías <1 %. |
| `modalidad_de_contratacion` | 15 | Igual; 10 categorías <1 %. |
| `tipodocproveedor` | 10 | One-hot; 8 categorías <1 %. Conservar el tipo, no la identidad. |
| `p_unidad_de_duracion` | 6 | One-hot u ordinal; `SIN_DATO`. |
| Binarias (`es_grupo`, `es_pyme`, etc.) | 2 | 0/1; no escalar. |

- Para one-hot: `OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)`. Así las categorías raras se agrupan y las no vistas en test no rompen.
- Para HGB: `OrdinalEncoder` con tope de 200 categorías (como en la F0), SIN categorías nativas: la app explica con SHAP TreeExplainer, que da atribuciones incorrectas con categorías nativas y por eso la app se niega a usarlas.
- `documentos_tipo` tiene "No" y "No definido" como categorías distintas: **no fusionarlas**. Solo unificar variantes de escritura ("No definido"/"No Definido") entre columnas.
- `nit_entidad` tiene 76,59 % de las filas en entidades con <1 % de participación: por eso no va con one-hot.

## 4. Escalado y transformaciones

### kNN y SVM

- `log_precio_base` y `log_pago_adelantado` **ya vienen en log1p**: no aplicar log otra vez; solo `RobustScaler`.
- Conteos (`p_numero_de_lotes`, `p_proveedores_invitados`, `p_respuestas_al_procedimiento`, `p_respuestas_externas`, `p_conteo_de_respuestas_a_ofertas`): `log1p` → `RobustScaler`.
- `p_dur_dias`: `<= 0` → NaN, recorte `[1, 3.650]`, `log1p` → `RobustScaler`.
- `len_objeto`, `dias_firma_a_inicio`, `dias_publicacion_a_firma`: `RobustScaler`.
- `mes_firma` y `dia_semana` son cíclicos: usar `sin`/`cos` o one-hot; **no** tratarlos como ordinales. `firma_ultimo_trimestre` y `fin_planeado_cruza_anio` son 0/1.
- Flags (`sin_proceso`, `sin_inicio`, `sin_duracion`, `precio_no_positivo`, `tiene_pago_adelantado`, `tiene_lotes`, `inicio_retroactivo`, `publicacion_tardia`): 0/1, sin escalar.

### HGB

- Valores crudos; HGB maneja faltantes. No escalar.
- Aplicar el recorte de duración `[1, 3.650]`; `log_precio_base` ya está en log.
- Códigos ordinales; `nit_entidad` con TargetEncoder.

## 5. Correlación entre numéricas: redundancia y efecto en kNN/SVM

Calculada con las 17 numéricas no constantes (se excluye `p_proveedores_que_manifestaron`). La matriz completa y el heatmap están en §3.5 del notebook.

| Par | Spearman | Pearson | Lectura | Acción |
|---|---:|---:|---|---|
| `p_proveedores_invitados` / `p_proveedores_con_invitacion` | **0,994** | 0,055 | Son casi la misma columna: iguales en 99,66 % de las filas y la segunda nunca es mayor. | kNN/SVM: conservar `p_proveedores_invitados`, eliminar `p_proveedores_con_invitacion`. HGB: mantener o probar sin una en J3. |
| `p_respuestas_al_procedimiento` / `p_proveedores_unicos_con` | **0,992** | 0,983 | Iguales en 99,48 % de las filas. | kNN/SVM: conservar `p_respuestas_al_procedimiento`, eliminar `p_proveedores_unicos_con`. HGB: mantener o ablación. |
| `mes_firma` / `firma_ultimo_trimestre` | **0,695** | **0,722** | El flag es una transformación del mes. | No usar ambos como ordinales. Para distancia: flag + mes cíclico, o solo uno. Con la v1 el Spearman es 0,695, justo bajo el umbral de 0,7; se mantiene la recomendación porque la bandera se deriva del mes. |
| `dias_firma_a_inicio` / `dias_publicacion_a_firma` | 0,066 | **−0,458** | No son redundantes: el Pearson lo producen los valores extremos, no el grueso. | Mantener ambas con `RobustScaler`; si la ablación se vuelve inestable, probar sin una. |
| `log_precio_base` / `len_objeto` | 0,148 | 0,064 | Sin problema. | Mantener. |

**Qué implica para kNN y SVM.** La colinealidad no rompe el modelo, pero **duplica el peso de la misma señal** en la distancia euclidiana y en el kernel: dos columnas casi idénticas cuentan como dos votos. Por eso se deja un representante de cada par redundante. Para HGB, que parte el espacio variable por variable, la colinealidad importa menos; se pueden mantener las dos y decidir con la ablación de J3. `p_proveedores_que_manifestaron` se elimina en todos los modelos porque es constante 0.

## 6. Deriva temporal

**Deriva 2019 → 2025 (base v1).**

| Señal | 2019 | 2025 | Cambio |
|---|---:|---:|---:|
| Proveedor con cédula | 75,55 % | 84,85 % | +9,29 pp |
| Proveedor con NIT | 23,61 % | 14,68 % | −8,93 pp |
| Contratación directa | 59,57 % | 75,59 % | +16,02 pp |
| Prestación de servicios | 79,59 % | 85,60 % | +6,02 pp |
| UNSPSC segmento 80 | 62,57 % | 72,29 % | +9,72 pp |
| Mínima cuantía | 15,08 % | 4,83 % | −10,25 pp |

- **No usar `anio_firma` como feature.** La deriva es fuerte y el modelo no vería años futuros; el año se usa solo para partir train/test.
- Ajustar imputación, escalado y codificación **solo con train**; validar temporalmente.
- One-hot con `handle_unknown="ignore"`; TargetEncoder con media global para entidades no vistas; HGB con códigos ordinales que toleren categorías nuevas.
- 2025 es test de deriva: no se usa para elegir hiperparámetros ni para ajustar el preprocesamiento.
- **Nota para J3:** la deriva 2019→2025 es real con la v1 (contratación directa 59,6 % → 75,6 %) y la tasa de y cambia entre train 2019-2023 (~28-30 %) y test 2024 (24,7 %). Además del train 2019-2023, probar una ventana de entrenamiento más reciente.

## 7. Madurez y censura

- **99,2 %** de la v1 es madura; 3.455 inmaduros = 11 sin duración + 867 con duración 0 + 2.577 con fin planeado + 60 días después del corte.
- Filtrar `eval_madura = True` en train y evaluación. `eval_*` no son features.
- 2025 tiene 97,64 % de madurez: aun así, se filtra.
- La etiqueta v1 es la de ampliación del plazo. Las decisiones de preprocesamiento que dependan de y se validan solo con train (2019-2023), nunca con 2024 o 2025.

## 8. Privacidad del proveedor

- Excluir identidad y datos del proveedor (documento, nombre, `codigo_proveedor`, NIT adjudicado). Conservar solo `tipodocproveedor`.
- **84,25 %** de los contratos tiene proveedor con cédula: es dato personal (Ley 1581 de 2012). No crear historial por proveedor, ni hash, ni ranking de personas.
- Esta es una decisión de J3/app, pero J2 no debe meter el identificador en el `Pipeline`.

## 9. Receta mínima del `Pipeline`

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   RobustScaler, TargetEncoder)
import numpy as np

from src import features as F

def limpiar_duracion(x):
    x = x.copy()
    x[x <= 0] = np.nan
    return np.clip(x, None, 3650)

num_log = ["p_numero_de_lotes", "p_proveedores_invitados",
           "p_respuestas_al_procedimiento", "p_respuestas_externas",
           "p_conteo_de_respuestas_a_ofertas"]
num_scaled = ["log_precio_base", "log_pago_adelantado", "len_objeto",
              "dias_firma_a_inicio", "dias_publicacion_a_firma"]
num_flags = ["firma_ultimo_trimestre", "fin_planeado_cruza_anio",
             "precio_no_positivo", "tiene_pago_adelantado"]
cat_onehot = [c for c in F.FEATURES_CAT if c != "nit_entidad"]

X = X.copy()
X["precio_no_positivo"] = (X["log_precio_base"] == 0).astype(int)
X["tiene_pago_adelantado"] = (X["log_pago_adelantado"] > 0).astype(int)

dur_pipe = Pipeline([
    ("limpiar", FunctionTransformer(limpiar_duracion)),
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", RobustScaler()),
])
num_log_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", RobustScaler()),
])
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", RobustScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="SIN_DATO")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,
                             sparse_output=False)),
])
flag_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0,
                              add_indicator=True)),
])

preprocesador = ColumnTransformer([
    ("dur", dur_pipe, ["p_dur_dias"]),
    ("num_log", num_log_pipe, num_log),
    ("num", num_pipe, num_scaled),
    ("flags", flag_pipe, num_flags),
    ("cat", cat_pipe, cat_onehot),
    ("entidad", TargetEncoder(cv=5, smooth="auto", target_type="binary"),
     ["nit_entidad"]),
], remainder="drop")
```

El bloque de arriba es la variante kNN/SVM. Para HGB, agrega de nuevo `p_proveedores_con_invitacion` y `p_proveedores_unicos_con` a `num_log`, cambia `cat_pipe` por `OrdinalEncoder` (tope 200), quita el `RobustScaler` de los numéricos y usa `HistGradientBoostingClassifier(categorical_features=None)`: los códigos ordinales se tratan como numéricos. No declarar máscara de categóricas (eso activa las categorías nativas que la app no puede explicar). Para `mes_firma`/`dia_semana`, agrega un transformador cíclico antes del escalado.

La limpieza de `p_dur_dias` (`<= 0` → NaN y recorte a 3.650) ya está dentro del `Pipeline` (`dur_pipe`), así que la app no tiene que replicarla. Es una regla fija, no aprendida.

## 10. Checklist para J2

- [ ] Filtrar `eval_madura = True` y separar train 2019–2023 / test 2024 / test 2025.
- [ ] Imputar con mediana/moda + indicadores; no borrar filas.
- [ ] `p_dur_dias`: `<= 0` → NaN y recorte `[1, 3.650]`.
- [ ] `precio_no_positivo` y `tiene_pago_adelantado`; no re-log de montos.
- [ ] `log1p` solo a conteos; `RobustScaler` para kNN/SVM.
- [ ] `mes_firma`/`dia_semana` cíclicos; flags 0/1.
- [ ] One-hot con `min_frequency=0.01` y `handle_unknown="ignore"`.
- [ ] `nit_entidad` con `TargetEncoder` dentro del `Pipeline`.
- [ ] Eliminar `p_proveedores_que_manifestaron`; para kNN/SVM, también `p_proveedores_con_invitacion` y `p_proveedores_unicos_con`.
- [ ] HGB: códigos ordinales, sin escalado.
- [ ] Sin identidad del proveedor, sin `anio_firma`, sin `eval_*`, sin `y*`.
- [ ] Re-ejecutar con la v1 y reajustar el TargetEncoder; el resto de acciones no cambia.

## 11. Qué no hacer

- No borrar contratos con faltantes ni con duración ≤ 0.
- No imputar los ceros de competencia como faltantes.
- No usar `anio_firma`, `eval_*` ni `y*` como features.
- No hacer one-hot de `nit_entidad`.
- No usar la identidad del proveedor.
- No ajustar imputación, escalado ni codificación con 2024/2025.
- Las tasas de la v1 estiman la población (muestra aleatoria del 10 % por dígito 4, con su margen de error), pero no usar las de 2024/2025 para decidir el preprocesamiento.
