# Reglas del proyecto — Riesgo de prórroga en contratos públicos (SECOP II)

Reglas compartidas por todo el equipo (Kevin, Jose y Geronimo) y por los agentes de código que usemos (Claude Code, opencode, Codex). Todos las seguimos; se cambian solo por acuerdo del equipo.

## Contexto

**Qué construimos:** un modelo que, al momento de firmar un contrato público, estima la probabilidad de que termine prorrogado. Lo usan veedurías y oficinas de control interno para priorizar qué contratos vigilar. Se despliega con una interfaz para personas sin conocimientos técnicos.

**Antes de cualquier tarea, lee lo que aplique:**
- `docs/Requeriments_FinalP.pdf`: enunciado del profesor (requisitos 3.1 a 3.9 y entregables).
- `docs/01_definicion_problema.md`: definición del problema, usuarios, etiqueta y datos.
- `00_exploracion/REPORTE_EXPLORACION.md` y `00_exploracion/NOTAS_DECISIONES.md`: hallazgos y decisiones de la exploración.

**Al reportar resultados**, marca cada afirmación: **[VERIFICADO]** (visto en los datos o el código), **[SUPUESTO]**, **[ESTIMADO]** o **[NO EVALUADO]**. Un "no se pudo" es mejor que un resultado inventado.

## Estructura del proyecto (propuesta, se confirma en el plan)

```
final-project/
├── README.md                  # guía del proyecto y cómo ejecutarlo
├── model_development.ipynb    # ENTREGABLE: EDA, preprocesamiento, entrenamiento y justificación
├── requirements.txt
├── docs/                      # enunciado, definición del problema, reglas, decisiones, lista de fuga
├── 00_exploracion/            # histórico de la exploración, solo lectura
├── src/                       # código reutilizable: descarga, etiqueta, features, entrenamiento
├── app/                       # interfaz desplegada
└── data/                      # no se versiona: raw/ (inmutable) y processed/
```

El notebook importa funciones de `src/` cuando el código es largo; en el notebook queda lo que hay que explicar.

## Git

**Ramas:**
- Nadie trabaja directamente en `master`.
- Una rama por tarea, con el formato `tipo/descripcion-corta` (ej. `feat/descarga-secop`, `fix/etiqueta-censura`).
- Antes de crear una rama, actualiza `master` (`git pull`) y parte de ahí.

**Commits (Conventional Commits):**
- Título en **inglés**, en imperativo, máximo 72 caracteres: `tipo(alcance): descripción`.
- Descripción en **español**: qué cambia y por qué.
- Tipos: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `data` (cambios en el pipeline de datos), `perf`.
- Commits pequeños y con un solo propósito. Cada quien hace commits con su propia identidad git.

```
feat(data): add SECOP contracts download with monthly cache

Descarga los contratos firmados entre 2019 y 2025 por la API de datos.gov.co,
con caché en parquet por mes para no repetir descargas. Excluye las columnas
con datos personales.
```

**Pull requests:**
- Título en inglés con la misma convención del commit.
- Descripción en español con estas secciones: **Qué cambia**, **Por qué**, **Cómo se verificó**, **Requisito del enunciado que cubre** (3.1–3.9) y **Pendientes**.
- Todo PR lo revisa otro integrante antes del merge (squash merge). Nadie hace merge de su propio PR sin revisión.
- Excepción: los PRs de cimientos y documentación de Kevin (F0 y docs) se mergean con la aprobación general del equipo, sin revisión formal.

**Nunca se versionan:** datos (`data/`), credenciales o tokens, ni archivos pesados generados. Antes de cada commit, verifica con `git status` que no se cuele nada de eso. **Excepción:** `app/respaldo_demo.parquet`: unos 40 contratos públicos, solo columnas de entrada, sin datos personales, para que la app desplegada responda si la API de datos.gov.co cae. Aprobado por Kevin el 26-oct.

## Entorno

- Usa el entorno virtual del repo y agrega a `requirements.txt` toda librería nueva (las librerías nuevas se acuerdan con el equipo).
- En scripts sin interfaz, usa `matplotlib.use("Agg")` para que `plt.show()` no se quede esperando una ventana.
- Ejecuta los notebooks con `jupyter nbconvert --execute` para que los outputs queden guardados.
- Requests a APIs: timeout de 30 s, máximo 3 reintentos, caché en disco por lote y log de progreso con tiempo transcurrido y ETA.
- Si algo va a tardar más de 10 minutos (descargas, entrenamientos, búsquedas de hiperparámetros), estima el tiempo antes de lanzarlo.

## Datos

**Fuente:** datos.gov.co (API Socrata, sin clave). Licencia CC BY-SA 4.0: hay que citar la fuente en el README y en el notebook.

| Conjunto | ID | Uso |
|---|---|---|
| Contratos Electrónicos | `jbjy-vk9h` | Variables del contrato y etiqueta |
| Procesos de Contratación | `p6dx-8zbt` | Variables del proceso de selección |
| Modificaciones a contratos | `u8cx-r425` | Mejora de la etiqueta |
| Adiciones | `cb9c-h8sn` | Descartado como etiqueta |
| Ejecución Contratos | `mfmm-jqmq` | Descartado: es fuga |

- **Dominios nuevos:** lee el `robots.txt` primero y en un paso separado. Ninguna otra request a ese dominio hasta evaluarlo.
- **Nunca inventes nombres de columnas:** verifícalos contra la metadata (`https://www.datos.gov.co/api/views/<ID>.json`).
- **Datos personales:** nunca se descargan las columnas con datos personales (representantes legales, ordenadores del gasto, supervisores, cuentas bancarias, documentos). La lista está en `00_exploracion/secop_descarga.py` (`PII`).
- `data/raw/` es inmutable: toda transformación se hace con código y se guarda en `data/processed/`.
- Antes de descargar un conjunto completo o un periodo grande, estima filas, tiempo y tamaño.

## Reglas de Machine Learning (no negociables)

1. **Momento de la predicción = fecha de firma.** Solo se usan variables disponibles en ese momento. La lista de fuga vive en `docs/` y se actualiza con cada hallazgo. Fuga ya confirmada: `estado_contrato`, `fecha_de_fin_del_contrato` y sus derivados, `valor_del_contrato` (incluye las adiciones), pagos, saldos y pendientes, `ultima_actualizacion`, liquidación, estados actuales del proceso, visualizaciones del proceso y todo lo que venga de Ejecución, Adiciones o Modificaciones.
2. **Etiqueta:** la definida en `docs/01_definicion_problema.md`. Cualquier cambio se discute en el equipo y se documenta.
3. **Split temporal:** train 2019-2023, test 2024 y 2025 como test de deriva. 2026 queda excluido (censura). Nunca uses split aleatorio. La validación dentro del train también es temporal.
4. **Sin fuga en el preprocesamiento:** todo se ajusta solo con train, dentro de un `Pipeline` de scikit-learn. El target encoding va dentro de cada fold. El early stopping usa validación temporal o se desactiva.
5. **Compara siempre contra las reglas simples:** tasa base y tasa histórica por entidad. Un modelo que no les gana con claridad no se acepta.
6. **Métricas:** PR-AUC (principal), ROC-AUC, precisión y recall en los primeros *k* contratos, siempre junto a la tasa base.
7. **Modelos del curso:** kNN, árboles, Random Forest, Gradient Boosting (`HistGradientBoostingClassifier`), SVM, SHAP para explicabilidad, y clustering y PCA como complemento. Otras librerías (LightGBM, XGBoost, etc.) solo con acuerdo del equipo y justificación.
8. **Reproducibilidad:** semilla fija (`42`) en todo lo aleatorio.
9. **Toda decisión se justifica en el notebook** con este formato: **Decisión**, **Alternativas consideradas**, **Por qué** y **Evidencia**. El profesor califica la justificación (requisitos 3.5 a 3.7).

## Convenciones de `model_development.ipynb`

- Una sección por requisito del enunciado (3.1 a 3.9), en ese orden.
- Markdown en español, corto y claro. Código simple, en el estilo de las clases (pandas y scikit-learn), sin sobreabstraer.
- Cada gráfica lleva un párrafo de interpretación: qué se observa y qué decisión motiva.
- El notebook debe correr de principio a fin sin errores.
