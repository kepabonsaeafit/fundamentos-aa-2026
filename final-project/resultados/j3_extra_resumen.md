# J3 extra: ablacion de `el_contrato_puede_ser_prorrogado` y contraste con fuga (6 montos y fuga evidente)

Complementa `resultados/j3_resumen.md`. Train 2019-2023, test 2024, solo maduros. IC 95 % con bootstrap por entidad (200 repeticiones). [VERIFICADO: salida de src/evaluar_j3_extra.py]

## 1. Ablacion de `el_contrato_puede_ser_prorrogado`

La feature esta excluida de FEATURES (seccion 0 punto 10 del plan). Aqui se mide el efecto de AGREGARLA al HGB final. En el train, 20.3% de los contratos trae "Si".

| Variante | PR-AUC | dif. vs regla | ROC-AUC | ROC-AUC intra (pares) |
|---|---|---|---|---|
| regla_entidad | 0,470 [0,355; 0,562] | +0,000 [0,000; 0,000] | 0,682 [0,628; 0,726] | 0,500 [0,500; 0,500] |
| final | 0,584 [0,488; 0,659] | +0,114 [0,082; 0,158] | 0,781 [0,752; 0,807] | 0,701 [0,597; 0,757] |
| final_mas_prorrogable | 0,583 [0,487; 0,658] | +0,113 [0,083; 0,155] | 0,781 [0,752; 0,808] | 0,702 [0,600; 0,757] |

Delta por agregar la columna: PR-AUC -0.0010, ROC-AUC +0.0005, ROC-AUC intra +0.0006.

Agregar `el_contrato_puede_ser_prorrogado` al HGB final cambia el PR-AUC 2024 en -0.0010 y el ROC-AUC en +0.0005 (dentro del ruido entre folds del modelo). Confirma la decision de dejarla fuera de FEATURES: no aporta senal util y es sospechosa de editarse despues de la firma.

## 2. Contraste con fuga sobre la base v1_A

Las columnas de fuga se leyeron de `data/raw/f2/contratos` solo para este contraste: 84 archivos, 452.515 filas, 21 duplicados por id_contrato eliminados (21 ids, copias exactas) y 0 contratos de la v1 sin match. La union vive solo en memoria: no se guarda en la base ni entra en FEATURES.

Train 261.249 contratos, test 88.088 (tasa base 0,247); regla por entidad con m = 10. IC 95 % con bootstrap por entidad (200 repeticiones).

| Variante | PR-AUC | dif. vs regla | ROC-AUC | dif. vs regla |
|---|---|---|---|---|
| regla_entidad | 0,470 [0,355; 0,562] | +0,000 [0,000; 0,000] | 0,682 [0,628; 0,726] | +0,000 [0,000; 0,000] |
| final_limpio | 0,584 [0,488; 0,659] | +0,114 [0,082; 0,158] | 0,781 [0,752; 0,807] | +0,099 [0,067; 0,130] |
| final_con_6_montos | 0,851 [0,810; 0,884] | +0,381 [0,318; 0,474] | 0,917 [0,901; 0,929] | +0,235 [0,200; 0,275] |
| final_con_fuga_evidente | 0,967 [0,957; 0,976] | +0,497 [0,412; 0,604] | 0,984 [0,980; 0,988] | +0,303 [0,262; 0,354] |

Al agregar los 6 montos al HGB limpio: PR-AUC +0.267, ROC-AUC +0.136. Al agregar la fuga evidente (estado_contrato one-hot, log1p de valor_del_contrato y dias de fecha_de_firma a fecha_de_fin_del_contrato, fechas truncadas al dia): PR-AUC +0.383, ROC-AUC +0.204.

### F0, base v0, y_dias (solo referencia historica)

Estas cifras no son de la v1 y no se usan como resultado; quedan para comparar con la F0.

[REFERENCIA HISTORICA: F0, base v0, y_dias; no se usa como resultado de la v1]

| Montaje | ROC-AUC | PR-AUC |
|---|---|---|
| HGB limpio (39 features) | 0.751 | 0.333 |
| HGB limpio + 6 montos | 0.807 | 0.429 |

Al meterle los 6 montos al HGB limpio, el ROC-AUC sube +0.056 (de 0.751 a 0.807) y el PR-AUC +0.096 (de 0.333 a 0.429).

junto con el precio base, los 6 montos delatan las adiciones de valor: su suma es igual a valor_del_contrato en el 80,6 % de los contratos (cociente sobre el precio base 1,26 con adicion frente a 1,00 sin adicion). El 0,807/0,429 se parece a la variante C de la exploracion (0,809/0,436), probablemente inflada por lo mismo.

Fuente: PLAN.md seccion 0, puntos 9 y 10; muestra de la exploracion (base v0, y_dias), no la base v1.

