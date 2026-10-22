# Definición del problema: riesgo de prórroga en contratos públicos (SECOP II)

> **Borrador v0.1** — pendiente de revisión del equipo. Las cifras marcadas como *preliminares* salen de la exploración de factibilidad (`00_exploracion/REPORTE_EXPLORACION.md`) y se actualizarán con el modelo final.

## 1. Contexto

En Colombia, las entidades públicas publican su contratación en el SECOP II, la plataforma transaccional de Colombia Compra Eficiente. Una vez firmado, un contrato estatal puede modificarse: se le puede **prorrogar** (ampliar el plazo) o **adicionar** (aumentar el valor). La Ley 80 de 1993 permite estas modificaciones y limita las adiciones de valor al 50 % del valor inicial del contrato. Sin embargo, la doctrina y la jurisprudencia coinciden en que deben ser una herramienta **excepcional**, justificada por hechos que no podían preverse al planear el contrato.

En la práctica, el uso de prórrogas es frecuente. Un estudio sobre la Gobernación del Tolima (2017-2018), hecho con datos del propio SECOP, encontró que la entidad recurría de forma constante a adiciones y prórrogas y las asoció con fallas en el principio de planeación. En nuestra exploración, cerca del **12 %** de los contratos firmados en 2024 terminaron con días adicionados, y la tasa subió a cerca del **20 %** en 2025 (*preliminar*).

El problema es de escala: cada año se firman cientos de miles de contratos y quienes los vigilan (veedurías, oficinas de control interno, periodistas) no pueden revisarlos todos. Hoy las prórrogas se detectan **después** de que ocurren, cuando ya hay un retraso en la entrega de obras o servicios.

## 2. Pregunta y formulación como problema de Machine Learning

**Pregunta:** al momento de firmar un contrato, ¿qué tan probable es que termine siendo prorrogado?

| Elemento | Definición |
|---|---|
| Tipo de tarea | Clasificación binaria supervisada. La salida es una probabilidad (puntaje de riesgo), no una etiqueta sí/no. |
| Unidad de predicción | Un contrato electrónico de SECOP II. |
| Momento de la predicción | La fecha de firma. Solo se usa información disponible hasta ese momento (datos del contrato y de su proceso de selección). |
| Etiqueta (inicial) | El contrato tuvo días adicionados (> 0). |
| Horizonte | El desenlace ocurre meses después de la predicción, durante la ejecución del contrato. |

**Sobre la etiqueta.** La etiqueta inicial tiene una limitación conocida: en una submuestra, la mayoría de los contratos con prórroga aprobada en el registro de modificaciones figuraban con cero días adicionados, es decir, la etiqueta **subestima** los positivos. Está planeado mejorarla con el conjunto de Modificaciones de SECOP II. La etiqueta final se documentará en el notebook.

**Sobre el horizonte.** Como el desenlace se observa tiempo después de la predicción, solo se evalúa el modelo con contratos cuyo plazo ya venció. Los contratos todavía en ejecución tienen una etiqueta incompleta (censura) y se excluyen de la evaluación. Además, el desempeño se reportará por separado según la duración planeada del contrato, para verificar que el modelo también funciona cuando el desenlace está lejos de la fecha de firma.

## 3. ¿Por qué deberían los datos de la firma predecir una prórroga?

Si la prórroga refleja con frecuencia una planeación deficiente, las señales de esa planeación deberían ser visibles al firmar. De ahí salen las hipótesis que guiarán el EDA. Son **hipótesis por verificar**, no hallazgos:

| Hipótesis | Señal observable al firmar |
|---|---|
| H1. Procesos apresurados se prorrogan más | Tiempo entre la publicación del proceso y la firma |
| H2. Poca competencia se asocia con más prórrogas | Número de oferentes o respuestas en el proceso |
| H3. La modalidad influye | Contratación directa frente a modalidades competitivas |
| H4. La anualidad presupuestal presiona el plazo | Contratos firmados en el último trimestre del año |
| H5. Los plazos subestimados terminan ampliándose | Duración planeada frente a la duración típica para ese tipo de objeto |
| H6. Las entidades tienen hábitos | Tasa histórica de prórrogas de la entidad contratante |
| H7. El tipo de objeto importa | Obra, servicios, suministros (clasificación UNSPSC) |

## 4. ¿Por qué Machine Learning y no una regla simple?

La alternativa más simple sería asumir que cada entidad repite su hábito: si una entidad suele prorrogar, sus contratos son riesgosos. Esa regla ya captura parte de la señal. Un modelo que combina las variables del contrato y del proceso de selección la supera, **pero por un margen modesto**. Cifras sobre contratos firmados en 2024 (23.999 contratos, tasa base de 0,123), con el modelo entrenado solo con 2019-2023 y usando únicamente variables disponibles al firmar (*preliminar*; intervalos de confianza del 95 % por remuestreo de entidades):

| | ROC-AUC | PR-AUC |
|---|---|---|
| Regla: tasa histórica de la entidad | 0,714 | 0,277 |
| Modelo (Gradient Boosting sin ajustar) | 0,751 | 0,333 |
| Diferencia | +0,037 [+0,011, +0,069] | +0,056 [+0,003, +0,113] |

Una versión anterior de este documento citaba un ROC-AUC cercano a 0,81. Esa cifra incluía variables que resultaron ser fuga de información (los montos por fuente de recursos suman el valor final del contrato, que ya incluye las adiciones) y se descartó.

Lo que sabemos hasta ahora sobre dónde aporta el modelo, con sus límites:

- **Entidades que no vio en el entrenamiento** (235 entidades, 804 contratos, 55 prorrogados). La regla no tiene información sobre ellas y queda en ROC-AUC 0,5 por construcción. El modelo alcanza un ROC-AUC de 0,708 [0,584, 0,800] y un PR-AUC de 0,180 frente a una tasa base de 0,068. El intervalo es ancho porque son pocos contratos.
- **Contratos de mayor riesgo.** En el 1 % de contratos con mayor puntaje (240), el 65 % terminó prorrogado [51 %, 78 %], frente al 44 % de la regla simple [17 %, 53 %] (el 54 % si la regla se suaviza para entidades con pocos contratos). En el 5 % y el 10 % de mayor puntaje la diferencia con la regla no se distingue de cero.
- **Variables por verificar.** Parte de la ventaja depende de variables que aún se están verificando, entre ellas la fecha de inicio del contrato. Con solo las variables ya verificadas, la diferencia con la regla en ROC-AUC (+0,027 [-0,002, +0,060]) y en PR-AUC (+0,038 [-0,013, +0,099]) no es clara.

Estas cifras salen de una muestra del 4 % de los contratos, con la etiqueta inicial (que subestima los positivos) y sin ajustar el modelo. Se rehacen con la muestra del 10 %, la etiqueta mejorada y el modelo ajustado, y el notebook reporta el resultado final con honestidad, aunque la ventaja resulte pequeña. El valor del ML es cruzar señales de planeación que ninguna regla individual captura y poder puntuar contratos de entidades sin historial.

## 5. Usuarios y decisión que apoya

**Usuario principal:** veedurías ciudadanas y oficinas de control interno de las entidades públicas.

**Decisión que apoya:** con recursos limitados, decidir **qué contratos vigilar primero**. Si una veeduría puede revisar a fondo 20 contratos de una alcaldía, el modelo los ordena por riesgo para que empiece por los que tienen más probabilidad de prorrogarse. Así puede hacer seguimiento antes de que el retraso ocurra. Esto define qué métricas importan (sección 3.7 de los requisitos): más que la exactitud global, interesa qué tan bien acierta el modelo en los contratos de mayor riesgo, es decir, la precisión en los primeros *k* contratos y el PR-AUC.

**Usuarios secundarios (trabajo futuro):** periodistas de datos, y los equipos de planeación de las propias entidades, que podrían revisar antes de firmar si el diseño de un contrato se parece al de los que terminan prorrogados.

## 6. Cómo interactúa el usuario con el modelo

El usuario no necesita conocer las variables del modelo ni llenar formularios largos.

1. **Consulta por contrato:** el usuario pega el número del contrato o el enlace de SECOP II. La aplicación consulta los datos en la API pública y devuelve el puntaje de riesgo junto con los factores que más lo explican, en lenguaje claro (explicabilidad con SHAP).
2. **Consulta por entidad:** el usuario elige una entidad y obtiene sus contratos recientes ordenados por riesgo.

**Regla de consistencia entre entrenamiento y uso.** Si la aplicación consulta un contrato que ya está en ejecución, la API devuelve algunos campos actualizados después de la firma. Por ejemplo, la fecha de fin ya incluye las prórrogas y el estado puede decir "modificado". Esos campos son fuga de información. La aplicación debe usar **exclusivamente** las variables disponibles al firmar, las mismas del entrenamiento; la duración planeada se toma del proceso de selección y no de la fecha de fin del contrato.

## 7. Origen de los datos

Todos los datos provienen de Datos Abiertos Colombia (datos.gov.co), publicados por la Agencia Nacional de Contratación Pública – Colombia Compra Eficiente, con licencia **CC BY-SA 4.0**. Se acceden por la API pública (Socrata) sin necesidad de clave y se actualizan continuamente, lo que permite automatizar el reentrenamiento (requisito 3.9).

| Conjunto | ID | Uso en el proyecto |
|---|---|---|
| Contratos Electrónicos | `jbjy-vk9h` | Variables del contrato y etiqueta inicial |
| Procesos de Contratación | `p6dx-8zbt` | Variables del proceso de selección (competencia, tiempos, precio base) |
| Modificaciones a contratos | `u8cx-r425` | Mejora de la etiqueta (pendiente) |
| Adiciones | `cb9c-h8sn` | Solo exploración; descartado como etiqueta por mezclar eventos administrativos |
| Ejecución Contratos | `mfmm-jqmq` | Descartado: es información posterior a la firma (fuga) y cubre solo una fracción de los contratos |

**Periodo:** contratos firmados entre 2019 y 2025. Antes de 2019 el campo de días adicionados casi no se diligenciaba, y los contratos de 2026 se excluyen porque la mayoría aún no termina. No se usan las columnas con datos personales (representantes legales, supervisores, cuentas bancarias, entre otras).

## 8. Alcance y consideraciones éticas

- **El modelo estima riesgo de prórroga, no detecta irregularidades ni corrupción.** Una prórroga puede ser legítima cuando la causan hechos imprevisibles. La interfaz debe comunicarlo así.
- **Es una herramienta de priorización.** La decisión de revisar, cuestionar o actuar sobre un contrato siempre es de una persona.
- **Limitaciones conocidas:** la etiqueta es imperfecta (sección 2), la tasa de prórrogas cambió en 2025 (lo que exige monitoreo de la deriva, ver estrategia de reajuste) y el modelo se basa en lo que las entidades registran en SECOP II, que puede tener errores de diligenciamiento.

## Referencias

- Congreso de la República de Colombia. Ley 80 de 1993, Estatuto General de Contratación de la Administración Pública (art. 40, parágrafo).
- *Las adiciones y prórrogas como mecanismo excepcional frente a los principios de planeación y anualidad presupuestal.* Administración & Desarrollo, ESAP, Vol. 49, Nº 2, 2019. https://repositoriocdim.esap.edu.co/items/0acfc0ef-7ced-4080-9d08-c68bafd5dca2
- Agencia Nacional de Contratación Pública – Colombia Compra Eficiente. SECOP II: Contratos Electrónicos (`jbjy-vk9h`), Procesos de Contratación (`p6dx-8zbt`), Modificaciones a contratos (`u8cx-r425`), Adiciones (`cb9c-h8sn`), Ejecución Contratos (`mfmm-jqmq`). Datos Abiertos Colombia, datos.gov.co. Licencia CC BY-SA 4.0.
