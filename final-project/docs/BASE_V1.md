# Base v1: qué es, qué pasó y qué se decidió (26-oct-2026)

## Qué es
- `data/processed/v1_A/base_contratos.parquet`: 452.494 contratos firmados
  entre 2019 y 2025, muestra del 10 % por último dígito 4 del `id_contrato`,
  autoponderada (`peso_mes` = 1). El sha256 está en el `meta.json`.
- Construida con `python -m src.construir_v1 --modo A`. Código de descarga:
  commits 93f2f0a y 27a6cf9; construcción: 9f1ab78.
- Corte de la descarga: 26-oct-2026, 06:38 (hora de Colombia).
- Etiqueta: `y = y_dias` o `y_modif`. Tasa: 27,6 %.

## Incidentes y cómo se resolvieron
1. **Caída de la fuente de Contratos.** El 25-oct a las 00:48 el conjunto
   quedó con 1.000 filas (el sábado tenía 5,6 millones). Se preparó un plan
   de respaldo (B: la muestra de la exploración con pesos por mes). Contratos
   volvió antes de las 06:31 del 26-oct, así que se publicó A. B no se
   publicó; su código queda pendiente en una rama aparte.
2. **La muestra de la exploración (v0) no era proporcional.** Era
   estratificada por mes con asignación igual (2.000 contratos por mes) y
   aleatoria dentro de cada mes. Distorsionaba la composición por año: la
   contratación directa en 2019 era 24 % en la v0 y es 59,6 % en la v1.
3. **Duplicados.** 21 `id_contrato` aparecen repetidos como copias exactas
   dentro de la misma unidad mensual; se aceptan y se deduplican (commit
   27a6cf9). Procesos y Modificaciones también traen filas duplicadas
   exactas, que F3 elimina.
4. **Fechas con hora.** Desde la republicación del 26-oct, la fecha de firma
   y la de inicio traen hora. Se truncan al día (`_a_dia` en `features.py` y
   `base.py`). Sin eso, `dias_firma_a_inicio` bajaba un día en el 53,9 % de
   los contratos. En la v0 el cambio no altera ningún valor.

## Decisiones
- **Madurez:** un contrato es maduro si su fin planeado más 60 días ya pasó
  al corte. Si la duración planeada es 0 o falta, es maduro si su firma más
  365 días ya pasó. Resultado: 99,2 % maduro.
- **No maduros:** se excluyen del entrenamiento y de la evaluación; `train` y
  `test` se filtran por `eval_madura = True`. No se usan como negativos ni se
  imputan; quedan solo para el análisis descriptivo de censura.
- **Chequeos 11 y 12** (montos negativos y duración fuera de 1-3.650 días):
  no bloquean, salvo que su tasa supere el doble de la de la v0.
- **Chequeo 15** (tasa de `y` por año entre 19 y 26 %): falló, porque
  2020-2023 quedan entre 27,8 y 30,0 %. Se aceptó: la banda salió del paso 3,
  con unos 200 contratos por año (±6 puntos), y recalculadas al estilo del
  paso 3 las tasas de todos los años caen dentro de sus intervalos. La v1
  muestra diferencias reales entre años que el paso 3 no podía detectar.
- **Sin pesos al entrenar:** A es autoponderada.

## Cifras que esta base reemplaza
- "Sin diferencias entre años (p = 0,56)" del paso 3: con la v1 la tasa va
  de 22,2 % (2019) a 30,0 % (2023), y baja a 24,7 % en 2024 por censura.
- Cadena de fuga (test 2024, ROC-AUC): modelo limpio 0,781; con los 6
  montos 0,917; con la fuga evidente 0,984. En la F0 (v0, `y_dias`) era
  0,751 → 0,807.

## Pendientes conocidos
- Modo B de `descarga.py` (Contratos desde archivo local): sin commitear.
- `leer_token()` debe validar la longitud del token y registrar de dónde lo
  leyó, sin mostrar su valor.
