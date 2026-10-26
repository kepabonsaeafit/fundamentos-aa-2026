"""Arma model_development.ipynb (raiz de final-project/) con las secciones 3.1 a 3.9 en orden.

Integra los borradores de notebooks/ (Jose: 3.2, 3.4-3.5, 3.6, 3.7), agrega las secciones de Kevin (3.1, 3.3) y
de Geronimo (3.8, 3.9) como marcadores que apuntan a sus docs, y actualiza 3.6 con el contraste con fuga y la
ablacion de el_contrato_puede_ser_prorrogado (src/evaluar_j3_extra.py). El notebook corre de principio a fin.

Uso (desde final-project/):  python build_notebook.py
"""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
NB = RAIZ / "notebooks"


def md(texto):
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def code(texto):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": texto.splitlines(keepends=True)}


def cargar(nombre):
    return json.loads((NB / f"{nombre}.ipynb").read_text(encoding="utf-8"))["cells"]


def sin_nota_borrador(cells):
    """La primera celda de cada borrador es un titulo con la nota 'Borrador ... se integra en model_development'.
    En el notebook integrado esa nota sobra: se quita la linea del blockquote."""
    out = []
    for c in cells:
        if c["cell_type"] == "markdown":
            src = "".join(c["source"])
            src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("> Borrador"))
            c = md(src)
        out.append(c)
    return out


PORTADA = md(
    "# Riesgo de ampliación del plazo en contratos públicos (SECOP II)\n"
    "## `model_development.ipynb` — entregable integrado (requisitos 3.1 a 3.9)\n\n"
    "Modelo que, al firmar un contrato estatal, estima la probabilidad de que termine con el plazo ampliado "
    "(prórroga, suspensión o ajuste de fechas). Lo usan veedurías y oficinas de control interno para priorizar "
    "qué contratos vigilar.\n\n"
    "**Fuente de datos:** Datos Abiertos Colombia (datos.gov.co), Agencia Nacional de Contratación Pública – "
    "Colombia Compra Eficiente. Licencia CC BY-SA 4.0. Conjuntos: Contratos Electrónicos (`jbjy-vk9h`), "
    "Procesos de Contratación (`p6dx-8zbt`) y Modificaciones a contratos (`u8cx-r425`).\n\n"
    "**Cómo se arma y se corre.** Las secciones leen la base procesada v1 (`data/processed/v1_A/`) y los "
    "resultados precalculados de J1 y J3 (`resultados/j3_*.json` y `*.csv`), generados por `src/modelado.py`, "
    "`src/evaluar_j3.py` y `src/evaluar_j3_extra.py`. Con los flags `RECALCULAR = True` de 3.5 y 3.6 se vuelven a "
    "correr esas etapas (unos 12 y 9 minutos). Cada decisión sigue el formato **Decisión / Alternativas / Por qué / "
    "Evidencia** (reglas del proyecto).\n\n"
    "**Reparto (sección 3 del plan).** Kevin: 3.1 y 3.3. Jose: 3.2, 3.4, 3.5, 3.6, 3.7 e integración. "
    "Geronimo: 3.8 y 3.9. Este notebook lo integra Jose y lo revisa Kevin.\n\n"
    "---"
)

SEC_31 = md(
    "# 3.1 Definición del problema\n\n"
    "> **Dueño: Kevin.** El texto completo está en `docs/01_definicion_problema.md`; aquí va el resumen que "
    "entra al entregable. Kevin revisa y amplía esta sección.\n\n"
    "**Pregunta.** Al firmar un contrato, ¿qué tan probable es que termine con el plazo ampliado, después de lo "
    "pactado al firmar?\n\n"
    "| Elemento | Definición |\n|---|---|\n"
    "| Tipo de tarea | Clasificación binaria supervisada; la salida es una probabilidad (puntaje de riesgo). |\n"
    "| Unidad de predicción | Un contrato electrónico de SECOP II. |\n"
    "| Momento de la predicción | La fecha de firma. Solo información disponible hasta ese momento. |\n"
    "| Etiqueta | Ampliación del plazo: días adicionados > 0 **o** una modificación publicada que extendió el "
    "plazo más de un día después de la firma. |\n"
    "| Horizonte | El desenlace ocurre meses después; solo se evalúan contratos maduros (plazo ya vencido). |\n\n"
    "**Usuarios y decisión.** Veedurías ciudadanas y oficinas de control interno. Con recursos limitados, deciden "
    "**qué contratos vigilar primero**: el modelo los ordena por riesgo. Por eso importan la precisión en los "
    "primeros *k* contratos y el PR-AUC más que la exactitud global (sección 3.7).\n\n"
    "**Qué ingresa el usuario** (enlaza con 3.8): el usuario solo pega el número del contrato. La app consulta "
    "Contratos y Procesos en datos.gov.co, construye **solo** las variables disponibles al firmar (las mismas del "
    "entrenamiento) e ignora las columnas de la lista de fuga, aunque la API las devuelva actualizadas.\n\n"
    "| Lo que pega el usuario | Lo que consulta la app | Lo que construye | Lo que descarta y por qué |\n"
    "|---|---|---|---|\n"
    "| Número del contrato | Contratos (`jbjy-vk9h`) y Procesos (`p6dx-8zbt`) | Las 39 `FEATURES` de "
    "`src/features.py` (tipo, modalidad, calendario, precio base, duración planeada, competencia, entidad…) | "
    "`estado_contrato`, `fecha_de_fin_del_contrato`, `valor_del_contrato`, los 6 montos, pagos y saldos: son "
    "posteriores a la firma (`docs/LISTA_FUGA.md`). |\n\n"
    "**Ética y alcance.** El modelo estima riesgo de ampliación del plazo, no detecta corrupción. Es una "
    "herramienta de priorización; la decisión de actuar siempre es de una persona.\n\n"
    "---"
)

SEC_33 = md(
    "# 3.3 Análisis exploratorio (EDA)\n\n"
    "> **Dueño: Kevin.** El EDA profundo (H1-H7 con evidencia, etiqueta, censura, deriva, fuga y la decisión de "
    "privacidad del proveedor) lo desarrolla Kevin a partir de `00_exploracion/REPORTE_EXPLORACION.md` y la base "
    "v1. Aquí queda el marco que el modelado usa; Kevin completa las gráficas y la narrativa.\n\n"
    "**Por qué los datos de la firma deberían predecir una ampliación del plazo** (hipótesis que guían el EDA y "
    "que la sección 3.6 contrasta con SHAP):\n\n"
    "| Hipótesis | Señal observable al firmar |\n|---|---|\n"
    "| H1. Procesos apresurados | Tiempo entre la publicación del proceso y la firma |\n"
    "| H2. Poca competencia | Número de oferentes o respuestas en el proceso |\n"
    "| H3. La modalidad influye | Contratación directa frente a modalidades competitivas |\n"
    "| H4. Anualidad presupuestal | Contratos firmados en el último trimestre del año |\n"
    "| H5. Plazos subestimados | Duración planeada frente a la típica para ese objeto |\n"
    "| H6. Hábitos de la entidad | Tasa histórica de ampliaciones de la entidad contratante |\n"
    "| H7. Tipo de objeto | Obra, servicios, suministros (clasificación UNSPSC) |\n\n"
    "**Etiqueta, censura y deriva** (resumen; detalle en `docs/01_definicion_problema.md` §2): la etiqueta `y` "
    "marca ampliación del plazo; los contratos aún en ejecución se excluyen por censura (solo maduros); la tasa de "
    "días adicionados salta en 2025 por un efecto de registro, no de comportamiento, y con `y` la tasa de 2025 "
    "queda dentro del rango de los demás años.\n\n"
    "**Fuga.** La lista viva es `docs/LISTA_FUGA.md`. La sección 3.6 muestra el contraste \"con fuga\" (los 6 "
    "montos) para dimensionar cuánto inflan el resultado.\n\n"
    "---"
)

SEC_38 = md(
    "# 3.8 Despliegue\n\n"
    "> **Dueño: Geronimo.** La interfaz para personas sin conocimientos técnicos vive en `app/`, con su URL "
    "pública y las capturas. Geronimo completa esta sección (arquitectura, flujo y capturas).\n\n"
    "**Flujo previsto.** El usuario pega el número del contrato → la app consulta Contratos y Procesos en "
    "datos.gov.co → `src/features.py` construye **solo** las variables disponibles al firmar → el `Pipeline` del "
    "artefacto `models/modelo_prorroga_v1.joblib` puntúa → se muestra el nivel de riesgo (bajo/medio/alto) y el "
    "percentil (sección 3.7), con los factores que más explican el puntaje (SHAP, en lenguaje claro).\n\n"
    "**Consistencia entrenamiento–uso.** La app reproduce exactamente el preprocesamiento (está dentro del "
    "`Pipeline` guardado) y descarta los campos de la lista de fuga aunque la API los devuelva actualizados.\n\n"
    "---"
)

SEC_39 = md(
    "# 3.9 Estrategia de reajuste\n\n"
    "> **Dueño: Geronimo.** El monitoreo y la lógica campeón-retador viven en `src/monitoreo.py`. Geronimo "
    "completa esta sección (cómo se detecta la degradación, el criterio, la frecuencia, cómo se valida y se "
    "promueve una nueva versión).\n\n"
    "**Base medida en la sección 3.6 (deriva 2025).** El modelo congelado 2019-2023 baja de 0,584 a 0,550 de "
    "PR-AUC en 2025 pero le sigue ganando a la regla; reentrenar con 2019-2024 no mejora el PR-AUC (0,550 en los "
    "dos casos). Por eso el artefacto v1 se queda en 2019-2023 y el reentrenamiento se decide con el protocolo "
    "campeón-retador: se reentrena con datos nuevos, se compara el retador contra el campeón en una ventana "
    "temporal reciente con las mismas métricas de 3.7, y solo se promueve si mejora de forma clara. "
    "`src/modelo.py: construir_pipeline()` es el que usa ese monitoreo para reentrenar.\n\n"
    "---"
)

# ---------------------------------------------------------------- 3.6: contraste con fuga + ablacion ampliada
FUGA_MD = md(
    "## Contraste \"con fuga\": los 6 montos de fuentes de recursos\n\n"
    "El enunciado pide mostrar que el modelo no tiene fuga. Para dimensionar cuánto *inflaría* el resultado una "
    "fuga, se mide cuánto sube el HGB si se le meten los **6 montos por fuente de recursos** "
    "(`presupuesto_general_de_la_nacion_pgn`, `sistema_general_de_participaciones`, `sistema_general_de_regal_as`, "
    "recursos propios de alcaldías/gobernaciones, `recursos_de_credito`, `recursos_propios`). Son **fuga "
    "confirmada**: su suma es igual a `valor_del_contrato` en el 80,6 % de los contratos, y ese valor ya incluye "
    "las adiciones (`docs/LISTA_FUGA.md`, 'Solo EDA')."
)

FUGA_CODE = code(
    'fuga = json.loads((RES / "j3_extra_fuga_montos.json").read_text(encoding="utf-8"))\n'
    'c = fuga["contraste_verificado_F0"]\n'
    'print("Los 6 montos NO están en la base v1 (son fuga; quedan solo en data/raw/ para el EDA):")\n'
    'print("  montos presentes en la base v1:", fuga["montos_presentes_en_la_base_v1"] or "ninguno")\n'
    'print("  archivos en data/raw/:", fuga["archivos_en_data_raw"] or "ninguno")\n'
    'print()\n'
    'print(c["estado"])\n'
    'display(pd.DataFrame({\n'
    '    "HGB limpio (39 features)": c["hgb_limpio_39_features"],\n'
    '    "HGB limpio + 6 montos":    c["hgb_con_6_montos"],\n'
    '}).T[["roc_auc", "pr_auc"]])\n'
    'print(f"Al agregar los 6 montos: ROC-AUC {c[\'subida_al_agregar_montos\'][\'roc_auc\']:+.3f}, "\n'
    '      f"PR-AUC {c[\'subida_al_agregar_montos\'][\'pr_auc\']:+.3f}")'
)

FUGA_INTERP = md(
    "**Interpretación:** al meterle los 6 montos al HGB limpio, el ROC-AUC sube de 0,751 a 0,807 (+0,056) y el "
    "PR-AUC de 0,333 a 0,429 (+0,096). Junto con el precio base, los montos delatan las adiciones de valor, que son "
    "exactamente lo que no se puede conocer al firmar. Ese 0,807/0,429 se parece a la variante C de la exploración "
    "(0,809/0,436), que estaba inflada por lo mismo. **Por eso los 6 montos quedan fuera de `FEATURES` y el modelo "
    "limpio reporta las cifras de 3.6.**\n\n"
    "> Este contraste se midió en la F0 con la muestra de la exploración (base v0). No se reconstruye aquí porque "
    "los 6 montos, al ser fuga, no están en la base v1 (solo en `data/raw/` para el EDA de fuga). "
    "`src/evaluar_j3_extra.py` deja el camino listo para medirlo sobre la base v1 si el crudo con esas columnas "
    "está disponible. [VERIFICADO en la F0; estado en `resultados/j3_extra_fuga_montos.json`]"
)

# Ablacion ampliada: reemplaza la celda de codigo y la interpretacion del borrador de 3.6.
ABL_CODE = code(
    'print("Ablación obligatoria:", F.FEATURES_ABLACION_J3)\n'
    'print(f"Por verificar ({len(F.FEATURES_POR_VERIFICAR)}):", F.FEATURES_POR_VERIFICAR)\n'
    'abl = pd.read_csv(RES / "j3_ablacion_2024.csv")\n'
    'display(tabla(abl, ["pr_auc", "roc_auc_intra_pares"], ["regla_entidad", "final", "sin_ablacion_j3", "sin_por_verificar"]))\n'
    '\n'
    '# el_contrato_puede_ser_prorrogado (excluida de FEATURES): efecto de AGREGARLA al HGB final\n'
    'ext = json.loads((RES / "j3_extra_resumen.json").read_text(encoding="utf-8"))["prorrogable"]\n'
    'prr = pd.read_csv(RES / "j3_extra_prorrogable_2024.csv")\n'
    'display(tabla(prr, ["pr_auc", "roc_auc_intra_pares"], ["regla_entidad", "final", "final_mas_prorrogable"]))\n'
    'print(f"Delta al agregar el_contrato_puede_ser_prorrogado: "\n'
    '      f"PR-AUC {ext[\'delta_pr_auc_por_agregar_prorrogable\']:+.4f}, "\n'
    '      f"ROC-AUC {ext[\'delta_roc_auc_por_agregar_prorrogable\']:+.4f}, "\n'
    '      f"ROC-AUC intra {ext[\'delta_roc_auc_intra_pares_por_agregar_prorrogable\']:+.4f}")'
)

ABL_INTERP = md(
    "**Interpretación:** sin las dos features de la fecha de inicio la ventaja sobre la regla casi no cambia "
    "(PR-AUC de +0,114 a +0,110; ROC-AUC dentro de entidad de +0,201 a +0,198) y los intervalos siguen lejos de 0. "
    "Quitar las 8 por verificar tampoco cambia nada. **La conclusión no depende de las variables con riesgo de "
    "fuga.**\n\n"
    "`el_contrato_puede_ser_prorrogado` (excluida de `FEATURES` por ser sospechosa de editarse tras la firma, "
    "sección 0 punto 10 del plan): **agregarla** al HGB final mueve el PR-AUC 2024 en −0,0010 y el ROC-AUC en "
    "+0,0005, dentro del ruido entre folds. Confirma la decisión de dejarla fuera: no aporta señal útil y sí "
    "riesgo de fuga. [VERIFICADO: `src/evaluar_j3_extra.py`]\n\n"
    "**Decisión:** se mantienen las 39 `FEATURES` y `el_contrato_puede_ser_prorrogado` queda fuera.\n"
    "- **Alternativas:** quitar las dos de la ablación o las 8 por verificar por precaución; o incluir "
    "`el_contrato_puede_ser_prorrogado`.\n"
    "- **Por qué:** ninguna de esas variantes cambia la conclusión; la limitación de la fecha de inicio (no ve "
    "cambios anteriores a la primera modificación) se cita como limitación.\n"
    "- **Evidencia:** las dos tablas de arriba."
)


def seccion_36():
    """Borrador de 3.6, con la ablacion ampliada y el contraste con fuga insertado antes del SHAP."""
    cells = sin_nota_borrador(cargar("06_generalizacion"))
    # cells: ... [11] md ablacion, [12] code ablacion, [13] md interp, [14] md SHAP, [15] code SHAP, [16] md interp
    # Reemplazo la celda de codigo de ablacion (12) y su interpretacion (13); inserto el contraste con fuga antes del SHAP (14).
    out = []
    for i, c in enumerate(cells):
        if i == 12:
            out.append(ABL_CODE)
        elif i == 13:
            out.append(ABL_INTERP)
        elif i == 14:
            out += [FUGA_MD, FUGA_CODE, FUGA_INTERP, c]   # contraste con fuga, luego el encabezado de SHAP
        else:
            out.append(c)
    return out


def main():
    cells = [PORTADA, SEC_31]
    cells += sin_nota_borrador(cargar("02_preprocesamiento"))   # 3.2
    cells += [SEC_33]
    cells += sin_nota_borrador(cargar("05_modelado"))           # 3.4 y 3.5
    cells += seccion_36()                                       # 3.6 (con fuga y ablacion ampliada)
    cells += sin_nota_borrador(cargar("07_metricas"))           # 3.7
    cells += [SEC_38, SEC_39]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    salida = RAIZ / "model_development.ipynb"
    salida.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Escrito {salida} con {len(cells)} celdas "
          f"({sum(c['cell_type'] == 'code' for c in cells)} de código, "
          f"{sum(c['cell_type'] == 'markdown' for c in cells)} de markdown)")


if __name__ == "__main__":
    main()
