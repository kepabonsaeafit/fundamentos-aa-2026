# Notas y decisiones de la exploracion (insumo para REPORTE_EXPLORACION.md)

Documento de trabajo. Se incorpora al reporte final en la Fase 3.

## SECOP II (checkpoint 2, decisiones del 2026-10-15)

1. **Anio de test.** Test principal = **2024** (train 2019-2023). El baseline del notebook `exploracion_secop.ipynb`
   se corrio con test = 2025 (train 2019-2024) porque esa era la salida de la regla de censura; **no se recalcula ahora**.
   Esas cifras (ROC-AUC 0,73 / PR-AUC 0,43 con features seguras de Contratos) quedan como **"test de drift"**:
   modelo entrenado con 2019-2024 evaluado sobre 2025. En la tabla de estabilidad del notebook, el test 2024 da
   ROC-AUC 0,76 / PR-AUC 0,33 con tasa base 0,123 (train 2019-2023).
   Evidencia de cambio en los datos: la tasa de `dias_adicionados > 0` sube de ~12 % (2023-24) a **20 % en 2025**
   en la poblacion completa (24 % en 2026), en todas las modalidades; no es un cambio de composicion.
   La causa no se pudo verificar [SUPUESTO: cambio en como se diligencia el campo o en las practicas].
2. **Etiqueta de tiempo.** Se sigue con `dias_adicionados > 0`. **Limitacion conocida**: subestima los positivos.
   En una submuestra de 2.286 contratos con filas en Modificaciones, el 59 % de los contratos con una
   modificacion Aprobada/Publicada con `dias_extendidos > 0` tiene `dias_adicionados = 0`
   (497 de 844; los 347 restantes si coinciden). Supuesto: `dias_extendidos` mide una prorroga efectiva.
3. **Conteo mensual de poblacion**: se omite (no se persistio al muestrear).
4. **Validacion interna.** `HistGradientBoostingClassifier` con `early_stopping='auto'` reserva un 10 % ALEATORIO del
   train como validacion. **Pendiente para el proyecto final**: la validacion interna debe ser temporal
   (p. ej. `early_stopping=False` o validacion manual por ano/mes).
5. **`y_adic_valor` se descarta como etiqueta.** No es estacionaria: la proporcion de contratos con un evento
   `Adicion en el valor` cae de 16-18 % (2018-19) a 7 % (2020) y a 2-3 % (2021-2026). Refleja como registran las
   entidades (mas uso de `Modificacion General`), no un cambio real. Con train 2019-2023 / test 2024 la tasa base
   pasa de ~7,3 % (promedio de las tasas anuales de la muestra 2019-2023) a 2,3 %. `y_cualquiera` (tiempo OR valor) hereda el problema y tambien se descarta.
   Tampoco hay una etiqueta de valor fiable en otra parte: `valor_modificacion` (Modificaciones) es el valor de la
   version del contrato, no el monto adicionado (igual a `valor_del_contrato` en el 75 % de las filas).

## Estimacion (sin descargar) del costo de bajar Modificaciones para los 211.997 contratos

Base: submuestra de 3.000 contratos (2022-24): 13.684 filas, 2.286 contratos con filas (76 %),
4,56 filas por contrato pedido, ~71 bytes por fila en CSV con 6 columnas, mediana 4 y p99 38 versiones por contrato.

| Concepto | Estimado |
|---|---|
| Lotes de 150 IDs | 1.414 |
| Filas crudas | ~970 mil (4,8 veces el tope de ~200k de la fase de exploracion) |
| Tamano CSV | ~100 MB |
| Tiempo, crudo | ~8-25 min con 3 hilos (supone 1-3 s por lote; no se midio el tiempo en la submuestra) |
| Alternativa agregada en el servidor (`GROUP BY id_contrato`, max/min de `valor_modificacion` y `dias_extendidos` por estado) | ~150-210 mil filas (dentro del tope), ~10-30 min (el `GROUP BY` costo ~2x en Adiciones) |
| Referencia de tiempos medidos en esta exploracion (segundos-hilo por lote) | Procesos 0,28; Ejecucion 0,34; Adiciones (agregado) 0,58 |

Riesgo: Modificaciones tiene ~30 M de filas y los `count(*)` directos ya agotaron el tiempo; un lote lento podria
disparar los reintentos. Requiere aviso previo (puede pasar de 10 min). La decision de descargar se toma despues de
comparar con Aire.

Lo que habilitaria: una etiqueta de tiempo mas completa (`max(dias_extendidos) > 0` en versiones aprobadas) y una
etiqueta de valor (valor de la ultima version vs. la primera) que no depende de como se tipifiquen los eventos.

## Aire Valle de Aburra: acceso a datos (Fase 2, 2026-10-15) - DETENIDO, esperando decision

| Fuente | Resultado | Estado |
|---|---|---|
| (a) datos.gov.co, catalogo | 9 busquedas: **ningun dataset de AMVA/SIATA** (la busqueda "SIATA" da 0). Si hay SISAIRE del IDEAM: `g4t8-zkc3` "Calidad del Aire en Colombia" | parcial |
| (a) SISAIRE `g4t8-zkc3` | Horario, nacional, 14 autoridades; incluye **AMVA** (estaciones de Medellin, dpto. `Antioquia`). PM2.5: 3,7 M filas a nivel nacional. Rango **2020-01-01 a 2024-12-31** (actualizado 2026-02-20; sin datos de 2025-26). Socrata, sin key. Columnas: estacion_id, nombre_fgda, nombre_est, msfl_code, med_concentracion_estandar, med_fecha_inicio/final, latitud, longitud, altitud, municipio, duraci_n | [VERIFICADO] automatizable, desactualizado |
| (b) siata.gov.co | Sin `robots.txt` (404). La seccion "Descargar informacion" muestra "Pagina deshabilitada... temporalmente". El unico enlace de descarga es de un portal con login (`descarga_siata/.../login/...`). Publica informes diarios en PDF. No se hallo API ni CSV | [VERIFICADO] bloqueo (login/deshabilitado) |
| (b') datos.siata.gov.co (Dataverse 6.10.1) | Dataset "Material Particulado - PM2.5", DOI 10.83041/AUWZWT, v4.0 publicada 2026-04-24, actualizada 2026-08-25. **162 archivos mensuales** `Estaciones_PM2.5_AAAA_MM.tab`, de 2013-01 a 2026-06, **14,6 MB en total**, los 162 con `restricted: false`. Pero `robots.txt` dice `User-agent: * / Disallow: /` y los terminos de uso remiten a un documento aparte (no leido) | **requiere decision**: robots.txt lo prohibe |
| (c) OpenAQ v3 | Una sonda sin credenciales: **401 "A valid API key must be provided"** | **bloqueo** (key); no se solicito |

Notas de proceso:
- Se leyo el robots.txt de datos.siata.gov.co y, en el mismo script, ya se habian hecho 2 GET de metadata
  (`/api/info/version` y la ficha del dataset) sin condicionarlos a ese resultado. No se descargo ningun archivo de datos.
  No se hace ninguna otra consulta a ese dominio hasta decidir.
- Del dataset de SIATA, lo verificado es: existencia, 162 archivos, rango 2013-01 a 2026-06 (por el nombre del
  primero y el ultimo), 0 restringidos, tamano total. NO verificado: numero de estaciones, columnas, resolucion
  (horaria o no), unidades, huecos. Una pagina de terceros menciona 16 estaciones PM2.5; sin confirmar.
- Fuentes de terceros mencionan endpoints "geoportal" de SIATA de tiempo real (72 h); no se probaron.

### Actualizacion: dataset SIATA descargado manualmente por el usuario (2026-10-15)
Ruta: `data/raw/siata_pm25/` (zip, `extraido/` con 162 CSV + MANIFEST.TXT, `docs/` con el PDF de terminos).
Inventario local (sin red) [VERIFICADO]:
- 162 CSV mensuales, 2013-01 a 2026-06, sin meses faltantes; 14,6 MB. Formato ancho: `fecha_hora` + una columna por estacion.
- 118.296 filas horarias = exactamente las horas del rango (0 horas sin fila, 0 duplicados).
- 33 codigos de estacion en total; 23-26 con datos desde 2018 (2013: 5, 2014: 6, 2015: 9, 2016: 8, 2017: 20).
  Hay estaciones que dejan de reportar (p. ej. MED-UNNV hasta 2020-02, CAL-LASA hasta 2022-05) y otras que empiezan
  (MED-PBLQ 2022, MED-PROV 2023-10, MED-CES 2024-04, EST-TABL 2024-05). Codigos de un solo dia o pocos meses: CAL-PMER, MED-PJIC, SAB-JOFE, GIR-TEMP.
- Validez dentro de la vida de cada estacion: ~89-98 % (excepto MED-SIAT 45 %). Valores: min 0, mediana 17, p99 64, max 232 ug/m3; sin negativos; 5.036 ceros (a revisar).
- No hay coordenadas de estaciones en estos archivos (hace falta para la meteorologia; con 2-3 puntos del valle basta para un spike).
- NO verificado: unidades (se asume ug/m3), si los valores ya estan validados por SIATA, criterio de agregacion horaria, significado de los codigos.
- PDF de terminos: NO se pudo leer (14 paginas; sin pdftoppm ni libreria PDF). Pendiente de leer.

## Aire Valle de Aburra: Fase 2 ejecutada (2026-10-15, checkpoint 3)
Notebook: `exploracion_aire.ipynb` (ejecutado con nbconvert, 0 errores). Resumen de terminos SIATA, calidad, estaciones y
baseline estan en el notebook. Notas para el reporte:
- Terminos SIATA (PDF "V1 - BORRADOR", abril 2026; licencia "XXXX" sin definir): uso academico permitido; atribucion
  obligatoria; prohibido uso comercial directo, alterar datos y entrenar IA/ML con fines comerciales sin permiso del AMVA;
  las descargas automatizadas deben coordinarse con el equipo tecnico de SIATA (11.2). Los datos no se suben al repo.
- Cita: Sistema de Alerta Temprana de Medellin y el Valle de Aburra (SIATA), AMVA. "Material Particulado - PM2.5" (v4.0).
  Repositorio de Datos SIATA. https://doi.org/10.83041/AUWZWT
- Test 2025-01 a 2026-06 no tiene ningun dia del promedio del valle > 37 ug/m3: el recall no se puede calcular para el objetivo principal.
- Oct-nov NO son criticos en estos datos; febrero-abril si (marzo: media 27,1).
- boundary_layer_height de Open-Meteo archive: faltan 181 dias (ene-jun 2024). Error mio corregido: primero lo atribui a 2026
  por aritmetica; los nulos por anio muestran que es 2024.

## Saber Pro: Fase 2B ejecutada (2026-10-15, checkpoint 4)
Notebook: `exploracion_saberpro.ipynb` (ejecutado con nbconvert, 0 errores). Datos en `data/raw/saberpro/` (no se suben al repo).
- Disponibilidad [VERIFICADO]: datos.gov.co `u37r-hjmu` "Resultados unicos Saber Pro": 1.217.482 filas, 57 columnas, periodos 2018-3 a 2022-6,
  actualizado 2023-08-23. NO hay 2023-2025. Licencia CC BY-SA 4.0 (atribucion al ICFES + compartir igual).
- El conjunto unificado NO trae `punt_global`. En 2019-2 (`2x55-9wxm`) el global = media simple de los 5 modulos (R2 0,9999, dif. max 0,4).
  Se asume igual en los demas periodos [SUPUESTO]. Objetivo del baseline = global reconstruido.
- Portal ICFES: robots.txt `Disallow: /` para todos. Se hizo GET a 2 paginas ANTES de gatear por robots.txt (no se leyo ni guardo su contenido).
  No se verifico si DataIcfes exige registro (terceros dicen que el FTP antiguo si). No se crearon cuentas.
- Cruce Saber 11 <-> Saber Pro [VERIFICADO con 4 filas de cada uno]: ids de formato distinto (EK... vs SB11...), sin llave comun.
  Cruzar por fecha de nacimiento + genero + colegio seria reidentificar: no se hizo.
- No se descargo `estu_fechanacimiento` ni `estu_privado_libertad` (la edad no es feature). Se excluyeron 314 filas sin estado PUBLICAR o sin los 5 puntajes.
- Baseline (test 2022, train 2018-3..2021-2, 47.560 filas): HGB MAE 17,2 / R2 0,31; media del train MAE 21,5 / R2 -0,01; media por institucion MAE 18,4 / R2 0,22.
  Split aleatorio (optimista): MAE 15,7 / R2 0,38. Domina la institucion (+3,6 MAE al permutar), no el estrato.
- Costo: muestra ~2 min de descarga, pico de RAM ~585 MB; dataset completo estimado ~18 min de descarga y ~3,5 GB como texto.

## SECOP: regla por entidad, entidades no vistas y HGB sin NIT (2026-10-15, local, sin red)
Notebook: `exploracion_secop.ipynb`, seccion 11 (reejecutado completo con nbconvert, 0 errores; las filas HGB A reproducen la seccion 9).
- Regla = tasa de `y_dias` por `nit_entidad` en el train, suavizada `(pos + m*p0)/(n + m)`; no vistas -> p0. `m` elegido con validacion temporal
  dentro del train (hasta T-2, valida T-1, ROC-AUC): m=0 (test 2024), m=1 (test 2025). ANOVA: rho 0,107 -> m ~8. Poco sensible (+-0,01 de ROC-AUC en el test).
- Test 2024 (train 2019-23): regla 0,714 / 0,277 (2,26x); HGB A 0,763 / 0,334 (2,73x); HGB C 0,809 / 0,436 (3,55x).
- Test 2025 (train 2019-24): regla 0,646 / 0,341 (1,74x); HGB A 0,727 / 0,427 (2,18x); HGB C 0,787 / 0,522 (2,67x).
- HGB - regla (bootstrap 95 %): A +0,048 [0,041, 0,056] (2024), +0,081 (2025); C +0,095 y +0,141.
- Entidades no vistas (muestra): 3,4 % (2024) y 4,6 % (2025) de los contratos; HGB A 0,674 y 0,790 de ROC-AUC (~55 y ~87 positivos, ruidoso); la regla 0,5 por construccion.
- Sin `nit_entidad`: A 0,753 (2024) y 0,736 (2025); C 0,825 y 0,800: no pierde senal. Sin entidad ni geografia: A 0,707 y 0,723.
- Limites: muestra ~4 % de la poblacion (la regla mejoraria con mas historial); HGB solo conoce las 200 entidades mas frecuentes por nombre; B y D no se calcularon con test 2024.
- Reporte actualizado: reajuste puntua solo obtener datos nuevos y reentrenar (SECOP 5, Aire 3, Saber Pro 1); totales SECOP 25, Aire 16, Saber Pro 16.
- Norma de 37 ug/m3 marcada [VERIFICADO indirectamente] en el reporte, segun el usuario (informes SIATA; Res. 2254/2017). El notebook de Aire conserva la etiqueta [SUPUESTO].
