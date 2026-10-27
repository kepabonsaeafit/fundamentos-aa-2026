# J3 extra: ablación de `valor_total_adjudicacion` / `valor_adjudicado_max` / `n_filas`

Corrida del 27-oct-2026 con el mismo protocolo de `src/evaluar_j3_extra.py`: HGB final v1, train 2019-2023, test 2024, solo maduros. Reproduce el `final` de `resultados/j3_extra_resumen.json` (PR-AUC 0,5839; ROC-AUC 0,7806), así que el protocolo es comparable. Script: `src/evaluar_j3_valor_adjudicado.py`.

## Por qué se excluyó de la v1

- `00_exploracion/exploracion_secop.ipynb` §4b y §8: con varios lotes/proveedores el valor adjudicado no es el valor del contrato; `valor_adjudicado_max` y `n_filas` quedan fuera por **ambigüedad de la medida**, no por fuga.
- El 90,39 % de los contratos tiene `valor_adjudicado_max = 0`; solo 43.458 tienen valor positivo. La columna es casi un indicador de "hubo adjudicación formal", y ese indicador no discrimina.
- De los 43.041 contratos con `fecha_adjudicacion` (9,51 % de la base), el 99,22 % se adjudicó el mismo día o antes de la firma y el 0,78 % después. La columna se conoce en la adjudicación, antes de la firma en la enorme mayoría; no hay evidencia de fuga temporal, pero tampoco se puede verificar con una sola foto.

## Resultado

| Variante | PR-AUC | ROC-AUC | ΔPR-AUC vs final [IC 95 %] | ΔROC-AUC vs final [IC 95 %] |
|---|---:|---:|---|---|
| final (39 features) | 0,5839 | 0,7806 | — | — |
| + log1p(valor_adjudicado_max) | 0,5840 | 0,7808 | +0,0002 [-0,0028; +0,0023] | +0,0002 [-0,0006; +0,0010] |
| + n_filas | 0,5839 | 0,7815 | +0,0001 [-0,0027; +0,0026] | +0,0009 [-0,0005; +0,0021] |
| + ambas | 0,5854 | 0,7813 | +0,0015 [-0,0032; +0,0041] | +0,0007 [-0,0005; +0,0017] |

IC 95 % con bootstrap por entidad (200 repeticiones). Todas las diferencias incluyen 0. Agregar la columna no aporta señal; se mantiene fuera de `FEATURES`.

## Calidad de la columna

- Cobertura del join con la base: 99,92 % (429.607 portafolios; 596.974 filas de Procesos).
- `valor_adjudicado_max = 0`: 90,39 % de los contratos.
- `n_filas > 1`: 1,79 % de los portafolios (7.702 de 429.607).
- Fechas: 43.041 contratos con `fecha_adjudicacion` (9,51 %); mediana −2 días respecto a la firma; p1 −137 días; p99 0 días.
