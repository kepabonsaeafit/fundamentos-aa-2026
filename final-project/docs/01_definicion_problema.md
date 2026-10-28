# Definición del problema: riesgo de ampliación del plazo en contratos públicos (SECOP II)

> **v1.0 (octubre de 2026).** Las cifras salen de las salidas ejecutadas de `model_development.ipynb` y de `resultados/`, con el modelo v1 y la base v1: una muestra del 10 % de los contratos firmados entre 2019 y 2025.
> Las cifras de la etiqueta salen de las Tablas 5.1 a 5.4 del notebook (sección 3.3.5); las de desempeño, de las secciones 3.6 y 3.7.

## 1. Contexto

En Colombia, las entidades públicas publican su contratación en el SECOP II, la plataforma transaccional de Colombia Compra Eficiente. Una vez firmado, un contrato estatal puede modificarse: se le puede **prorrogar** (ampliar el plazo) o **adicionar** (aumentar el valor). La Ley 80 de 1993 permite estas modificaciones y limita las adiciones de valor al 50 % del valor inicial del contrato. Sin embargo, la doctrina y la jurisprudencia coinciden en que deben ser una herramienta **excepcional**, justificada por hechos que no podían preverse al planear el contrato.

En la práctica, el uso de prórrogas es frecuente. Un estudio sobre la Gobernación del Tolima (2017-2018), hecho con datos del propio SECOP, encontró que la entidad recurría de forma constante a adiciones y prórrogas y las asoció con fallas en el principio de planeación. Con la etiqueta aprobada (sección 2), el 27,6 % de los contratos de la base v1 (452.494 contratos firmados entre 2019 y 2025) termina con el plazo ampliado: más de uno de cada cuatro. Por año de firma la tasa va de 22,2 % (2019) a 30,0 % (2023); en 2024 es 24,7 % y en 2025, 27,1 % (Tabla 5.1 del notebook), así que la etiqueta no tiene un salto en 2025. Los que sí saltan en 2025 son los días adicionados (de 12,1 % a 20,0 %), una razón más para no usarlos solos como etiqueta (sección 2).

El problema es de escala: cada año se firman cientos de miles de contratos y quienes los vigilan (veedurías, oficinas de control interno, periodistas) no pueden revisarlos todos. Hoy las ampliaciones del plazo se detectan **después** de que ocurren, cuando ya hay un retraso en la entrega de obras o servicios.

## 2. Pregunta y formulación como problema de Machine Learning

**Pregunta:** al momento de firmar un contrato, ¿qué tan probable es que termine con el plazo ampliado, es decir, después de lo pactado al firmar?

| Elemento | Definición |
|---|---|
| Tipo de tarea | Clasificación binaria supervisada. La salida es una probabilidad (puntaje de riesgo), no una etiqueta sí/no. |
| Unidad de predicción | Un contrato electrónico de SECOP II. |
| Momento de la predicción | La fecha de firma. Solo se usa información disponible hasta ese momento (datos del contrato y de su proceso de selección). |
| Etiqueta | Ampliación del plazo: el contrato tuvo días adicionados o una modificación publicada que extendió su plazo después de la firma (definición abajo). |
| Horizonte | El desenlace ocurre meses después de la predicción, durante la ejecución del contrato. |

**Sobre la etiqueta (aprobada el 23 de octubre de 2026).** Un contrato es positivo (`y = 1`) si tiene días adicionados (`dias_adicionados > 0`) **o** si en el conjunto de Modificaciones tiene una modificación publicada (estado `Publicado`) con días extendidos (`dias_extendidos > 0`) y aprobada más de un día después de la firma (`fecha_de_aprobacion > fecha_de_firma + 1 día`). El día de margen cubre un desfase de zona horaria entre los dos conjuntos.

- **Por qué no bastan los días adicionados.** La etiqueta inicial (`dias_adicionados > 0`) subestima los positivos. En la base v1 marca el 13,3 % de los contratos frente al 27,6 % de la etiqueta aprobada: solo el 48,2 % de los positivos tiene días adicionados, y los otros 64.759 contratos solo aparecen en Modificaciones (Tablas 5.1 y 5.4 del notebook).
- **Qué mide.** Que el contrato terminó después de lo pactado al firmar. Por eso el objetivo se llama **ampliación del plazo** y no prórroga en sentido estricto: las suspensiones y los ajustes de fecha también cuentan como positivos. Qué proporción de los positivos son de ese tipo no se midió en la base v1 [NO EVALUADO].
- **Limitaciones.** La etiqueta descansa casi entera en Modificaciones: coincide con la que usa solo Modificaciones salvo en 150 contratos que únicamente tienen días adicionados (Tabla 5.3). Una ampliación que no quede registrada ni en los días adicionados ni en Modificaciones no se marca, y su frecuencia no se midió [NO EVALUADO].

El notebook documenta la etiqueta en 3.3.5.

**Sobre el horizonte.** Como el desenlace se observa tiempo después de la predicción, solo se evalúa el modelo con contratos cuyo plazo ya venció. Los contratos todavía en ejecución tienen una etiqueta incompleta (censura) y se excluyen de la evaluación. Además, el desempeño por tramo de duración planeada (hasta 3, de 3 a 6, de 6 a 12 y más de 12 meses) está reportado en la sección 3.7 del notebook, para verificar que el modelo también funciona cuando el desenlace está lejos de la fecha de firma. El modelo le gana a la regla por entidad en todos los tramos.

## 3. ¿Por qué deberían los datos de la firma predecir una ampliación del plazo?

Si la ampliación del plazo refleja con frecuencia una planeación deficiente, las señales de esa planeación deberían ser visibles al firmar. De ahí salen las siete hipótesis que guiaron el EDA; el notebook las contrasta con los datos en la sección 3.3.6:

| Hipótesis | Señal observable al firmar |
|---|---|
| H1. Los procesos apresurados terminan más con el plazo ampliado | Tiempo entre la publicación del proceso y la firma |
| H2. Poca competencia se asocia con más ampliaciones del plazo | Número de oferentes o respuestas en el proceso |
| H3. La modalidad influye | Contratación directa frente a modalidades competitivas |
| H4. La anualidad presupuestal presiona el plazo | Contratos firmados en el último trimestre del año |
| H5. Los plazos subestimados terminan ampliándose | Duración planeada frente a la duración típica para ese tipo de objeto |
| H6. Las entidades tienen hábitos | Tasa histórica de ampliaciones del plazo de la entidad contratante |
| H7. El tipo de objeto importa | Obra, servicios, suministros (clasificación UNSPSC) |

## 4. ¿Por qué Machine Learning y no una regla simple?

La alternativa más simple sería asumir que cada entidad repite su hábito: si una entidad suele ampliar los plazos, sus contratos son riesgosos. Esa regla ya captura parte de la señal. El modelo, que combina las variables del contrato y del proceso de selección, la supera con claridad: la diferencia en PR-AUC es de +0,114 y su intervalo no incluye 0. Cifras sobre los 88.088 contratos de 2024 con el plazo ya vencido (tasa base de 0,247), con el modelo entrenado solo con 2019-2023 y usando únicamente variables disponibles al firmar (sección 3.6 del notebook; entre corchetes, intervalos de confianza del 95 % por remuestreo de entidades, 200 repeticiones):

| | ROC-AUC | PR-AUC |
|---|---|---|
| Regla: tasa histórica de la entidad | 0,682 [0,628; 0,726] | 0,470 [0,355; 0,562] |
| Modelo (HGB ajustado, v1) | 0,781 [0,752; 0,807] | 0,584 [0,488; 0,659] |
| Diferencia | +0,099 [+0,067; +0,130] | +0,114 [+0,082; +0,158] |

**Antecedente (no es un resultado de la v1).** Una versión anterior de este documento citaba un ROC-AUC cercano a 0,81, que incluía los montos por fuente de recursos: esos montos suman el valor final del contrato, que ya incluye las adiciones, y se descartaron por fuga. En la v1 esa fuga se mide: el ROC-AUC del modelo sube de 0,781 a 0,917 al agregarle esos 6 montos, y a 0,984 con la fuga evidente (estado, valor y fecha de fin del contrato; sección 3.6).

Dónde aporta el modelo, con sus límites (test 2024):

- **Entidades que no vio en el entrenamiento** (1.834 contratos, 211 con el plazo ampliado, tasa base de 0,115). La regla no tiene información sobre ellas y queda en ROC-AUC 0,5 por construcción. El modelo alcanza un ROC-AUC de 0,721 [0,655; 0,791], +0,221 sobre la regla [+0,155; +0,291], y un PR-AUC de 0,279 [0,170; 0,449]. El intervalo del PR-AUC es ancho porque son pocos positivos.
- **Dentro de una misma entidad.** Es lo que enfrenta una veeduría que revisa los contratos de una sola entidad. Entre un contrato con el plazo ampliado y uno sin ampliación de la misma entidad, el modelo puntúa más alto al primero en el 70 % de los pares (ROC-AUC dentro de entidad de 0,701 [0,604; 0,753]; 0,703 [0,691; 0,716] promediando por entidad; 1.199 entidades con ambas clases y 11.953.889 pares; sección 3.7). La regla simple no distingue entre contratos de una misma entidad (0,5 por construcción).
- **Contratos de mayor riesgo.** Si se revisa el 5 % de mayor puntaje de 2024 (unos 4.400 contratos), el 83 % termina con el plazo ampliado, frente al 74 % con la regla por entidad y al 25 % al azar; en el 1 % más alto son el 93 % frente al 84 %, y en el 10 %, el 73 % frente al 59 %. La ventaja sobre la regla solo se distingue de cero en el 10 % (+0,135 [+0,065; +0,199]); en el 1 % y en el 5 % el intervalo de la diferencia incluye el 0 (sección 3.7).
- **Variables con riesgo de fuga.** Sin las dos variables que usan la fecha de inicio, el PR-AUC de 2024 pasa de 0,584 a 0,580 (+0,110 sobre la regla [+0,079; +0,153]); sin las 8 variables por verificar queda en 0,584 (+0,114 [+0,083; +0,157]). La ventaja sobre la regla no depende de esas variables (sección 3.6). La limitación de la fecha de inicio está en `docs/LISTA_FUGA.md`.

El modelo supera a la regla por entidad en PR-AUC y en ROC-AUC, con más ventaja donde la regla no tiene información (entidades no vistas, dentro de una misma entidad) y menos en la parte más alta de la lista. En 2025 el modelo congelado baja a un PR-AUC de 0,550, pero sigue ganándole a la regla (+0,084 [+0,033; +0,157]; sección 3.6). El valor del ML es cruzar señales de planeación que ninguna regla individual captura y poder puntuar contratos de entidades sin historial.

## 5. Usuarios y decisión que apoya

**Usuario principal:** veedurías ciudadanas y oficinas de control interno de las entidades públicas.

**Decisión que apoya:** con recursos limitados, decidir **qué contratos vigilar primero**. Si una veeduría puede revisar a fondo 20 contratos de una alcaldía, el modelo los ordena por riesgo para que empiece por los que tienen más probabilidad de terminar con el plazo ampliado. Así puede hacer seguimiento antes de que el retraso ocurra. Esto define qué métricas importan (sección 3.7 de los requisitos): más que la exactitud global, interesa qué tan bien acierta el modelo en los contratos de mayor riesgo, es decir, la precisión en los primeros *k* contratos y el PR-AUC.

**Usuarios secundarios (trabajo futuro):** periodistas de datos, y los equipos de planeación de las propias entidades, que podrían revisar antes de firmar si el diseño de un contrato se parece al de los que terminan con el plazo ampliado.

## 6. Cómo interactúa el usuario con el modelo

El usuario no necesita conocer las variables del modelo ni llenar formularios largos.

**Consulta por contrato:** el usuario pega el número del contrato o el enlace de SECOP II. La aplicación consulta los datos en la API pública y devuelve el nivel de riesgo (bajo, medio o alto) y su percentil, junto con los factores que más lo explican, en lenguaje claro (explicabilidad con SHAP).

**Trabajo futuro: consulta por entidad.** El usuario elegiría una entidad y obtendría sus contratos recientes ordenados por riesgo. La aplicación actual solo consulta por contrato.

**Regla de consistencia entre entrenamiento y uso.** Si la aplicación consulta un contrato que ya está en ejecución, la API devuelve algunos campos actualizados después de la firma. Por ejemplo, la fecha de fin ya incluye las ampliaciones del plazo y el estado puede decir "modificado". Esos campos son fuga de información. La aplicación debe usar **exclusivamente** las variables disponibles al firmar, las mismas del entrenamiento; la duración planeada se toma del proceso de selección y no de la fecha de fin del contrato.

## 7. Origen de los datos

Todos los datos provienen de Datos Abiertos Colombia (datos.gov.co), publicados por la Agencia Nacional de Contratación Pública – Colombia Compra Eficiente, con licencia **CC BY-SA 4.0**. Se acceden por la API pública (Socrata) sin necesidad de clave y se actualizan continuamente, lo que permite automatizar el reentrenamiento (requisito 3.9).

| Conjunto | ID | Uso en el proyecto |
|---|---|---|
| Contratos Electrónicos | `jbjy-vk9h` | Variables del contrato y parte de la etiqueta (días adicionados) |
| Procesos de Contratación | `p6dx-8zbt` | Variables del proceso de selección (competencia, tiempos, precio base) |
| Modificaciones a contratos | `u8cx-r425` | Etiqueta: modificaciones publicadas que extienden el plazo |
| Adiciones | `cb9c-h8sn` | Solo exploración; descartado como etiqueta por mezclar eventos administrativos |
| Ejecución Contratos | `mfmm-jqmq` | Descartado: es información posterior a la firma (fuga) y cubre solo una fracción de los contratos |

**Periodo:** contratos firmados entre 2019 y 2025. Antes de 2019 el campo de días adicionados casi no se diligenciaba, y los contratos de 2026 se excluyen porque la mayoría aún no termina. No se usan las columnas con datos personales (representantes legales, supervisores, cuentas bancarias, entre otras).

## 8. Alcance y consideraciones éticas

- **El modelo estima riesgo de ampliación del plazo, no detecta irregularidades ni corrupción.** Una ampliación del plazo (prórroga, suspensión o ajuste de fechas) puede ser legítima cuando la causan hechos imprevisibles. La interfaz debe comunicarlo así.
- **Es una herramienta de priorización.** La decisión de revisar, cuestionar o actuar sobre un contrato siempre es de una persona.
- **Limitaciones conocidas:** la etiqueta es imperfecta (sección 2); los días adicionados saltan en 2025 (de 12,1 % a 20,0 %) pero la etiqueta no (de 24,7 % a 27,1 %), lo que se atribuye a cómo se guarda ese campo [SUPUESTO: la causa no se midió]; en 2025 el modelo congelado baja de PR-AUC 0,584 a 0,550 y sigue ganándole a la regla, y la deriva se monitorea (ver estrategia de reajuste); y el modelo se basa en lo que las entidades registran en SECOP II, que puede tener errores de diligenciamiento.

## Referencias

- Congreso de la República de Colombia. Ley 80 de 1993, Estatuto General de Contratación de la Administración Pública (art. 40, parágrafo).
- *Las adiciones y prórrogas como mecanismo excepcional frente a los principios de planeación y anualidad presupuestal.* Administración & Desarrollo, ESAP, Vol. 49, Nº 2, 2019. https://repositoriocdim.esap.edu.co/items/0acfc0ef-7ced-4080-9d08-c68bafd5dca2
- Agencia Nacional de Contratación Pública – Colombia Compra Eficiente. SECOP II: Contratos Electrónicos (`jbjy-vk9h`), Procesos de Contratación (`p6dx-8zbt`), Modificaciones a contratos (`u8cx-r425`), Adiciones (`cb9c-h8sn`), Ejecución Contratos (`mfmm-jqmq`). Datos Abiertos Colombia, datos.gov.co. Licencia CC BY-SA 4.0.
