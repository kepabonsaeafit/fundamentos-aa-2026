# EDA para preprocesamiento (J2) — base v0

> **Aviso:** Basado en la muestra de la exploración (v0). Pendiente del veredicto sobre su aleatoriedad dentro del año; las cifras de deriva podrían cambiar.

> **Para Jose. Fecha:** 25-oct-2026. **Base:** `data/processed/base_contratos.parquet` v0 (167.997 contratos, 2019–2025, ~24.000 por año). **Etiqueta:** `y = y_dias`, provisional.
>
> La v0 **no es proporcional por año** y su etiqueta subestima las prórrogas. Las acciones de esta nota dependen solo de las features (`X`), no de `y`, así que se mantienen con la v1. Las cifras de tasa o de relación con `y` no se usan aquí. El detalle reproducible está en `notebooks/03_eda.ipynb` §2, §3, §4 y §7.1.

## 0. Lo que necesita J2

- `construir_features` entrega **39 features**: 18 numéricas y 21 categóricas. No hay que modificar `src/features.py`; el `Pipeline` consume `X`.
- Filtrar **`eval_madura = True`** para entrenar y evaluar. `eval_*` nunca entra como feature.
- **Split temporal:** train 2019–2023, test 2024 y 2025 como test de deriva. Nada se ajusta con 2024/2025.
- Todo lo que aprende estadísticas (imputación, escalado, one-hot, target encoding) va **dentro del `Pipeline`** y se ajusta solo con train. El `TargetEncoder` de `nit_entidad` se ajusta **solo con el fold de entrenamiento**; las entidades no vistas reciben la media global.
- Elimina `p_proveedores_que_manifestaron`: es constante 0 (215 nulos).

## 1. Faltantes

| Bloque | n | % | Acción |
|---|---:|---:|---|
| Sin fila de Procesos (todas las `p_*`) | 215 | 0,13 | Imputar mediana/moda + `sin_proceso`. No imputar 0: 0 significa "sin respuesta". |
| `fecha_de_inicio_del_contrato` | 1.732 | 1,03 | `dias_firma_a_inicio` queda NaN: imputar mediana + `sin_inicio`. |
| Duración planeada (`p_dur_dias`) | 436 | 0,26 | NaN: imputar mediana + `sin_duracion`. |
| `p_fecha_de_publicacion_del` | 229 | 0,14 | `dias_publicacion_a_firma` NaN: imputar mediana + `sin_publicacion`. |
| `unspsc_familia` / `unspsc_segmento` | 1.375 | 0,82 | Categoría `SIN_UNSPSC` (todas son `UNSPECIFIED`); no borrar filas. |
| `p_unidad_de_duracion` | 215 | 0,13 | Categoría `SIN_DATO`. |
| `y_modif` | 167.997 | 100 | No es feature; se llena en la v1. |

**No borrar filas.** Los bloques son pequeños y borrarlos sesga la muestra. Los indicadores de faltante son parte de la señal: un contrato sin proceso no es lo mismo que uno con proceso y cero respuestas.

## 2. Valores imposibles y atípicos

| Variable | Hallazgo | Acción |
|---|---|---|
| `p_dur_dias` | 8.694 (5,18 %) ≤ 0; 66 (0,04 %) > 3.650 días; máximo 83,4 millones | `<= 0` → NaN; recortar a `[1, 3.650]`; `sin_duracion`; `log1p` solo para kNN/SVM. |
| `p_precio_base` | 26 negativos, 1.740 ceros, máximo 3,4 billones | `log_precio_base` ya viene en `log1p` con `clip(lower=0)`: los negativos quedan en 0 y no se separan de los ceros reales. Crear `precio_no_positivo = log_precio_base == 0`; no volver a aplicar log. |
| `valor_de_pago_adelantado` | 99,8 % en cero; máximo 38.500 millones; 49 "No" y 100 "No Definido" con valor > 0; 1 "Si" con valor 0 | Usar `log_pago_adelantado` (ya en `log1p`) + `tiene_pago_adelantado`; conservar `habilita_pago_adelantado`. No corregir a mano. |
| `p_numero_de_lotes` | 91,1 % en cero; máximo 244 | El cero es "sin lotes", no faltante. `tiene_lotes` + `log1p` para distancia. |
| Conteos de competencia | 79–99 % en cero | El cero es "sin respuesta/invitación", no faltante. No imputar; `log1p` para distancia; flags de cero si se quiere. |
| `p_proveedores_que_manifestaron` | Constante 0 | Eliminar en todos los modelos. |
| `len_objeto` | Máximo 500 caracteres; 1 cero | No es un problema de outliers; escalar para distancia. |
| `dias_firma_a_inicio`, `dias_publicacion_a_firma` | Ya vienen recortadas (`[-365, 365]` y `[-30, 1000]`) | No volver a recortar; crear `inicio_retroactivo` y `publicacion_tardia` si el modelo las aprovecha. |

No eliminar outliers: son pocos y pueden ser señal. Para HGB, usar el recorte suave de duración y dejar el resto crudo. Para kNN/SVM, `log1p` + `RobustScaler`.

## 3. Cardinalidad y codificación

| Columna | Categorías | Acción |
|---|---:|---|
| `nit_entidad` | 2.568 | **TargetEncoder** siempre dentro del `Pipeline`, ajustado solo con el fold de entrenamiento (`cv=5`, `smooth="auto"`, `target_type="binary"`); entidades no vistas → media global. No one-hot. Es la señal de H6. |
| `ciudad` | 664 | Agrupar <1 % en `OTROS` + one-hot; para HGB, códigos ordinales o categorías nativas. `No Definido` (23,5 %) es categoría, no faltante. |
| `unspsc_familia` | 373 | Igual que `ciudad`; `SIN_UNSPSC` como categoría. |
| `unspsc_segmento` | 56 | Igual; 48 categorías <1 %. |
| `justificacion_modalidad_de` | 39 | One-hot con agrupación <1 %; HGB nativo/ordinal. |
| `departamento` | 34 | Igual; `No Definido` (2,0 %) es categoría. |
| `sector` | 25 | Igual. |
| `tipo_de_contrato` | 23 | Igual; 17 categorías <1 %. |
| `modalidad_de_contratacion` | 14 | Igual; 7 categorías <1 %. |
| `tipodocproveedor` | 9 | One-hot; 7 categorías <1 %. Conservar el tipo, no la identidad. |
| `p_unidad_de_duracion` | 6 | One-hot o nativo; `SIN_DATO`. |
| Binarias (`es_grupo`, `es_pyme`, etc.) | 2 | 0/1; no escalar. |

- Para one-hot: `OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)`. Así las categorías raras se agrupan y las no vistas en test no rompen.
- Para HGB: `OrdinalEncoder` con tope de 200 categorías (como en la F0) o categorías nativas. J3 decide por el efecto en SHAP; J2 deja las dos variantes.
- `documentos_tipo` tiene "No" y "No definido" como categorías distintas: **no fusionarlas**. Solo unificar variantes de escritura ("No definido"/"No Definido") entre columnas.
- `nit_entidad` tiene 77,1 % de las filas en entidades con <1 % de participación: por eso no va con one-hot.

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
- Categorías nativas o códigos ordinales; `nit_entidad` con TargetEncoder.
- Para las dos variantes (con y sin categorías nativas), usar el mismo `Pipeline` cambiando solo el codificador.

## 5. Correlación entre numéricas: redundancia y efecto en kNN/SVM

Calculada con las 17 numéricas no constantes (se excluye `p_proveedores_que_manifestaron`). La matriz completa y el heatmap están en §3.5 del notebook.

| Par | Spearman | Pearson | Lectura | Acción |
|---|---:|---:|---|---|
| `p_proveedores_invitados` / `p_proveedores_con_invitacion` | **0,996** | 0,073 | Son casi la misma columna: iguales en 99,59 % de las filas y la segunda nunca es mayor. | kNN/SVM: conservar `p_proveedores_invitados`, eliminar `p_proveedores_con_invitacion`. HGB: mantener o probar sin una en J3. |
| `p_respuestas_al_procedimiento` / `p_proveedores_unicos_con` | **0,995** | 0,971 | Iguales en 99,04 % de las filas. | kNN/SVM: conservar `p_respuestas_al_procedimiento`, eliminar `p_proveedores_unicos_con`. HGB: mantener o ablación. |
| `mes_firma` / `firma_ultimo_trimestre` | **0,753** | 0,753 | El flag es una transformación del mes. | No usar ambos como ordinales. Para distancia: flag + mes cíclico, o solo uno. |
| `dias_firma_a_inicio` / `dias_publicacion_a_firma` | −0,101 | **−0,743** | No son redundantes: el Pearson lo producen los valores extremos, no el grueso. | Mantener ambas con `RobustScaler`; si la ablación se vuelve inestable, probar sin una. |
| `log_precio_base` / `len_objeto` | 0,12 | 0,04 | Sin problema. | Mantener. |

**Qué implica para kNN y SVM.** La colinealidad no rompe el modelo, pero **duplica el peso de la misma señal** en la distancia euclidiana y en el kernel: dos columnas casi idénticas cuentan como dos votos. Por eso se deja un representante de cada par redundante. Para HGB, que parte el espacio variable por variable, la colinealidad importa menos; se pueden mantener las dos y decidir con la ablación de J3. `p_proveedores_que_manifestaron` se elimina en todos los modelos porque es constante 0.

## 6. Deriva temporal

| Señal | 2019 | 2025 | Cambio |
|---|---:|---:|---:|
| Proveedor con cédula | 44,6 % | 82,4 % | +37,8 pp |
| Proveedor con NIT | 53,1 % | 17,1 % | −36,0 pp |
| Contratación directa | 24,2 % | 72,5 % | +48,3 pp |
| Prestación de servicios | 52,0 % | 82,9 % | +30,9 pp |
| UNSPSC segmento 80 | 30,7 % | 70,1 % | +39,4 pp |
| Mínima cuantía | 38,3 % | 6,0 % | −32,3 pp |

- **No usar `anio_firma` como feature.** La deriva es fuerte y el modelo no vería años futuros; el año se usa solo para partir train/test.
- Ajustar imputación, escalado y codificación **solo con train**; validar temporalmente.
- One-hot con `handle_unknown="ignore"`; TargetEncoder con media global para entidades no vistas; HGB con códigos ordinales que toleren categorías nuevas.
- 2025 es test de deriva: no se usa para elegir hiperparámetros ni para ajustar el preprocesamiento.
- **Nota para J3:** la deriva 2019→2025 afecta la validación temporal. Además del train 2019–2023, probar una ventana de entrenamiento más reciente (por ejemplo, 2021–2023 o 2022–2023) y comparar contra el mismo test.

## 7. Madurez y censura

- **99,2 %** de la v0 es madura; 1.360 inmaduros = 436 sin duración + 924 con fin planeado + 60 días después del corte.
- Filtrar `eval_madura = True` en train y evaluación. `eval_*` no son features.
- 2025 tiene 97,37 % de madurez: aun así, se filtra.
- La etiqueta v0 subestima las prórrogas; la v1 cambia la tasa. No ajustar decisiones de preprocesamiento con la tasa v0.

## 8. Privacidad del proveedor

- Excluir identidad y datos del proveedor (documento, nombre, `codigo_proveedor`, NIT adjudicado). Conservar solo `tipodocproveedor`.
- **74,6 %** de los contratos tiene proveedor con cédula: es dato personal (Ley 1581 de 2012). No crear historial por proveedor, ni hash, ni ranking de personas.
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

El bloque de arriba es la variante kNN/SVM. Para HGB, agrega de nuevo `p_proveedores_con_invitacion` y `p_proveedores_unicos_con` a `num_log`, cambia `cat_pipe` por `OrdinalEncoder` (tope 200) o categorías nativas, quita el `RobustScaler` de los numéricos y declara la máscara de categóricas en `HistGradientBoostingClassifier`. Para `mes_firma`/`dia_semana`, agrega un transformador cíclico antes del escalado.

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
- [ ] HGB: dos variantes (categorías nativas y códigos ordinales), sin escalado.
- [ ] Sin identidad del proveedor, sin `anio_firma`, sin `eval_*`, sin `y*`.
- [ ] Re-ejecutar con la v1 y reajustar el TargetEncoder; el resto de acciones no cambia.

## 11. Qué no hacer

- No borrar contratos con faltantes ni con duración ≤ 0.
- No imputar los ceros de competencia como faltantes.
- No usar `anio_firma`, `eval_*` ni `y*` como features.
- No hacer one-hot de `nit_entidad`.
- No usar la identidad del proveedor.
- No ajustar imputación, escalado ni codificación con 2024/2025.
- No interpretar las tasas de la v0 como población: la muestra no es proporcional por año y la etiqueta es provisional.
