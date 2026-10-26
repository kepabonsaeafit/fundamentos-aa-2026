"""Contrato central de features: una sola funcion usada en el entrenamiento y en la app.

Aqui va lo determinista y fila a fila. Todo lo que se aprende de los datos (imputacion, codificacion,
escalado, target encoding) va dentro del Pipeline de scikit-learn que se guarda con el modelo.
Solo se usan variables disponibles al firmar el contrato (momento de la prediccion = fecha de firma).
"""
import numpy as np
import pandas as pd

FEATURES_VERSION = "1"

# Fechas fuera de [2000, 2100) son errores de digitacion (en los datos hay anios 0025 y 8201)
FECHA_MIN = pd.Timestamp("2000-01-01")
FECHA_MAX = pd.Timestamp("2100-01-01")

# Columnas de entrada: lo unico que lee construir_features. Nombres de la API; las de Procesos llevan prefijo p_.
COLS_CONTRATO = [
    "fecha_de_firma", "fecha_de_inicio_del_contrato", "codigo_de_categoria_principal", "objeto_del_contrato",
    "tipo_de_contrato", "modalidad_de_contratacion", "justificacion_modalidad_de",
    "origen_de_los_recursos", "destino_gasto", "espostconflicto", "es_grupo", "es_pyme",
    "habilita_pago_adelantado", "valor_de_pago_adelantado", "documentos_tipo",
    "el_contrato_puede_ser_prorrogado", "tipodocproveedor",
    "nit_entidad", "sector", "orden", "rama", "entidad_centralizada", "departamento", "ciudad",
]
COLS_PROCESO = [
    "p_precio_base", "p_duracion", "p_unidad_de_duracion", "p_fecha_de_publicacion_del", "p_numero_de_lotes",
    "p_proveedores_invitados", "p_proveedores_con_invitacion", "p_proveedores_que_manifestaron",
    "p_respuestas_al_procedimiento", "p_respuestas_externas", "p_conteo_de_respuestas_a_ofertas",
    "p_proveedores_unicos_con",
]
ENTRADAS = COLS_CONTRATO + COLS_PROCESO

# Una fila por feature: (nombre, tipo, familia, hipotesis que apoya, estado frente a la fuga).
# "por verificar" = se asume que el campo no cambia despues de la firma, pero no esta comprobado (J3 hace la ablacion).
# "por verificar, excluida" = ademas es sospechosa, asi que no entra a FEATURES; se puede pedir con
# construir_features(df, con_excluidas=True) para evaluarla en la ablacion de J3.
EXCLUIDA = "por verificar, excluida"
# "segura con limitacion" = se verifico y no hay diferencia por y, pero el metodo no ve todos los casos
# (docs/LISTA_FUGA.md, seccion 6). Sigue siendo feature; J3 hace la ablacion.
SEGURA_CON_LIMITACION = "segura con limitacion"
_TABLA = [
    # ---- categoricas ----
    ("tipo_de_contrato", "cat", "Contrato", "H7", "seguro"),
    ("modalidad_de_contratacion", "cat", "Contrato", "H3", "seguro"),
    ("justificacion_modalidad_de", "cat", "Contrato", "H3", "seguro"),
    ("unspsc_segmento", "cat", "Contrato", "H7", "seguro"),
    ("unspsc_familia", "cat", "Contrato", "H7", "seguro"),
    ("origen_de_los_recursos", "cat", "Contrato", "-", "seguro"),
    ("destino_gasto", "cat", "Contrato", "-", "seguro"),
    ("espostconflicto", "cat", "Contrato", "-", "seguro"),
    ("es_grupo", "cat", "Contrato", "-", "seguro"),
    ("es_pyme", "cat", "Contrato", "-", "seguro"),
    ("habilita_pago_adelantado", "cat", "Contrato", "-", "seguro"),
    ("documentos_tipo", "cat", "Contrato", "-", "seguro"),
    ("el_contrato_puede_ser_prorrogado", "cat", "Contrato", "-", EXCLUIDA),
    ("tipodocproveedor", "cat", "Contrato", "-", "seguro"),      # solo el tipo de documento, no la identidad
    ("nit_entidad", "cat", "Entidad", "H6", "seguro"),           # base de la tasa historica (TargetEncoder)
    ("sector", "cat", "Entidad", "H6 (proxy)", "seguro"),
    ("orden", "cat", "Entidad", "H6 (proxy)", "seguro"),
    ("rama", "cat", "Entidad", "H6 (proxy)", "seguro"),
    ("entidad_centralizada", "cat", "Entidad", "H6 (proxy)", "seguro"),
    ("departamento", "cat", "Geografia", "H6 (proxy)", "seguro"),
    ("ciudad", "cat", "Geografia", "H6 (proxy)", "seguro"),
    ("p_unidad_de_duracion", "cat", "Proceso", "H5", "seguro"),
    # ---- numericas ----
    ("log_pago_adelantado", "num", "Contrato", "-", "por verificar"),
    ("len_objeto", "num", "Contrato", "H7 (proxy de complejidad)", "seguro"),
    ("dias_firma_a_inicio", "num", "Contrato", "-", SEGURA_CON_LIMITACION),
    ("mes_firma", "num", "Calendario", "H4", "seguro"),
    ("dia_semana", "num", "Calendario", "-", "seguro"),
    ("firma_ultimo_trimestre", "num", "Calendario", "H4", "seguro"),
    ("fin_planeado_cruza_anio", "num", "Calendario", "H4", SEGURA_CON_LIMITACION),
    ("log_precio_base", "num", "Proceso", "- (tamano del contrato)", "seguro"),
    ("p_dur_dias", "num", "Proceso", "H5", "seguro"),
    ("dias_publicacion_a_firma", "num", "Proceso", "H1", "seguro"),
    ("p_numero_de_lotes", "num", "Proceso", "- (complejidad)", "seguro"),
    ("p_proveedores_invitados", "num", "Proceso", "H2", "por verificar"),
    ("p_proveedores_con_invitacion", "num", "Proceso", "H2", "por verificar"),
    ("p_proveedores_que_manifestaron", "num", "Proceso", "H2", "por verificar"),
    ("p_respuestas_al_procedimiento", "num", "Proceso", "H2", "por verificar"),
    ("p_respuestas_externas", "num", "Proceso", "H2", "por verificar"),
    ("p_conteo_de_respuestas_a_ofertas", "num", "Proceso", "H2", "por verificar"),
    ("p_proveedores_unicos_con", "num", "Proceso", "H2", "por verificar"),
]
FEATURES_CAT = [f[0] for f in _TABLA if f[1] == "cat" and f[4] != EXCLUIDA]
FEATURES_NUM = [f[0] for f in _TABLA if f[1] == "num" and f[4] != EXCLUIDA]
FEATURES = FEATURES_CAT + FEATURES_NUM
FEATURES_POR_VERIFICAR = [f[0] for f in _TABLA if f[4] == "por verificar"]
FEATURES_EXCLUIDAS = [f[0] for f in _TABLA if f[4] == EXCLUIDA]
# Ablacion obligatoria de J3 (docs/LISTA_FUGA.md seccion 6): HGB final sin estas dos features, contra la regla, global y dentro de entidad.
# Va POR NOMBRE: no se deriva del estado en _TABLA, asi que no cambia si ese estado cambia. Debe ser un subconjunto de FEATURES.
FEATURES_ABLACION_J3 = ["dias_firma_a_inicio", "fin_planeado_cruza_anio"]
assert set(FEATURES_ABLACION_J3) <= set(FEATURES), "la ablacion de J3 nombra una feature que no entra al modelo"
_TIPO = {f[0]: ("object" if f[1] == "cat" else "float64") for f in _TABLA}

# Feature -> texto para el usuario de la app (las explicaciones SHAP se muestran con estos nombres)
NOMBRES_LEGIBLES = {
    "tipo_de_contrato": "Tipo de contrato",
    "modalidad_de_contratacion": "Modalidad de contratación",
    "justificacion_modalidad_de": "Justificación de la modalidad",
    "unspsc_segmento": "Segmento del bien o servicio (UNSPSC)",
    "unspsc_familia": "Familia del bien o servicio (UNSPSC)",
    "origen_de_los_recursos": "Origen de los recursos",
    "destino_gasto": "Destino del gasto",
    "espostconflicto": "Contrato de posconflicto",
    "es_grupo": "Contratista es un grupo",
    "es_pyme": "Contratista es una pyme",
    "habilita_pago_adelantado": "Permite pago adelantado",
    "documentos_tipo": "Usa documentos tipo",
    "el_contrato_puede_ser_prorrogado": "Contrato prorrogable según el registro",
    "tipodocproveedor": "Tipo de documento del contratista",
    "nit_entidad": "Entidad contratante",
    "sector": "Sector de la entidad",
    "orden": "Orden de la entidad",
    "rama": "Rama del poder público",
    "entidad_centralizada": "Entidad centralizada",
    "departamento": "Departamento",
    "ciudad": "Ciudad",
    "p_unidad_de_duracion": "Unidad en que se planeó la duración",
    "log_pago_adelantado": "Valor del pago adelantado",
    "len_objeto": "Longitud de la descripción del objeto",
    "dias_firma_a_inicio": "Días entre la firma y el inicio",
    "mes_firma": "Mes de la firma",
    "dia_semana": "Día de la semana de la firma",
    "firma_ultimo_trimestre": "Firmado en el último trimestre del año",
    "fin_planeado_cruza_anio": "El plazo planeado cruza el cambio de año",
    "log_precio_base": "Precio base del proceso",
    "p_dur_dias": "Duración planeada (días)",
    "dias_publicacion_a_firma": "Días entre la publicación del proceso y la firma",
    "p_numero_de_lotes": "Número de lotes del proceso",
    "p_proveedores_invitados": "Proveedores invitados",
    "p_proveedores_con_invitacion": "Proveedores con invitación directa",
    "p_proveedores_que_manifestaron": "Proveedores que manifestaron interés",
    "p_respuestas_al_procedimiento": "Respuestas al procedimiento",
    "p_respuestas_externas": "Respuestas de entes externos",
    "p_conteo_de_respuestas_a_ofertas": "Respuestas directas en las ofertas",
    "p_proveedores_unicos_con": "Proveedores únicos que redactaron respuestas",
}
assert set(NOMBRES_LEGIBLES) == set(FEATURES + FEATURES_EXCLUIDAS)

# Fuga confirmada (espejo de docs/LISTA_FUGA.md): nunca entran como entrada ni como feature.
# Los nombres de Procesos aparecen tambien con prefijo p_ (ver mas abajo).
_FUGA_CONTRATOS = {
    # estado y actualizaciones posteriores a la firma
    "estado_contrato", "ultima_actualizacion",
    # la fecha de fin incluye las prorrogas
    "fecha_de_fin_del_contrato", "fecha_de_notificaci_n_de_prorrogaci_n", "dur_contrato_dias", "razon_dur",
    # sospechosa (LISTA_FUGA 4): se actualiza en parte despues de la firma; tampoco es respaldo de la duracion en la censura
    "duraci_n_del_contrato",
    # el valor incluye las adiciones, y los 6 montos por fuente de recursos suman ese mismo valor
    "valor_del_contrato", "log_valor_contrato", "razon_valor",
    "presupuesto_general_de_la_nacion_pgn", "sistema_general_de_participaciones", "sistema_general_de_regal_as",
    "recursos_propios_alcald_as_gobernaciones_y_resguardos_ind_genas_", "recursos_de_credito", "recursos_propios",
    # pagos, saldos y pendientes (ejecucion)
    "valor_facturado", "valor_pagado", "valor_pendiente_de_pago", "valor_amortizado", "valor_pendiente_de",
    "valor_pendiente_de_ejecucion", "saldo_cdp", "saldo_vigencia", "dias_firma_a_ult_actualizacion",
    # liquidacion y reversion
    "liquidaci_n", "fecha_inicio_liquidacion", "fecha_fin_liquidacion",
    "reversion", "fecha_inicio_reversi_n", "fecha_fin_reversi_n",
}
_FUGA_PROCESOS = {
    "visualizaciones_del", "estado_del_procedimiento", "id_estado_del_procedimiento", "estado_resumen",
    "fase", "estado_de_apertura_del_proceso", "fecha_de_ultima_publicaci",
}
_FUGA_OTROS_DATASETS = {
    # Ejecucion
    "n_filas_ejec", "avance_real_max",
    # Adiciones
    "n_eventos", "n_adic_valor", "n_extension", "n_mod_general", "primer_evento", "ultimo_evento",
    # Modificaciones (las que definen la etiqueta o sirven para deduplicarla y ordenarla; LISTA_FUGA 3)
    "valor_modificacion", "dias_extendidos", "estado_modificacion", "numero_version",
    "fecha_de_aprobacion", "fecha_fin_contrato", "fecha_version", "identificador_modificacion",
    "version_anterior_del_contrato",
}
_ETIQUETAS = {"dias_adicionados", "y", "y_dias", "y_modif"}
COLUMNAS_FUGA = (_FUGA_CONTRATOS | _FUGA_PROCESOS | _FUGA_OTROS_DATASETS | _ETIQUETAS
                 | {"p_" + c for c in _FUGA_PROCESOS})


def tabla_features():
    """Tabla de revision: una fila por feature con su familia, la hipotesis que apoya y su estado de fuga."""
    return pd.DataFrame(_TABLA, columns=["feature", "tipo", "familia", "hipotesis", "estado_fuga"])


# ---------------------------------------------------------------- auxiliares (conversion de texto a tipos)
def _a_fecha(s):
    """Texto ISO (como lo devuelve la API) o fecha ya tipada -> datetime64; fuera de [2000, 2100) -> NaT."""
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors="coerce", format="ISO8601")
    return s.where((s >= FECHA_MIN) & (s < FECHA_MAX))


def _a_dia(s):
    """Como _a_fecha, pero truncada al dia (00:00). Desde la republicacion del 26-oct, Contratos trae
    hora en fecha_de_firma e inicio; la v0 y las features se definieron con fechas sin hora."""
    return _a_fecha(s).dt.normalize()


def _a_numero(s):
    return pd.to_numeric(s, errors="coerce")


def _a_texto(s):
    """Texto o faltante (NaN); nunca pd.NA, para que el dtype sea el mismo venga de donde venga.

    Una columna numerica (p. ej. nit_entidad, que en la muestra de entrenamiento esta tipada como numero y en la
    API llega como texto) se pasa a texto sin decimales: 800214750.0 -> '800214750'.
    """
    if pd.api.types.is_numeric_dtype(s):
        s = s.astype("float64").map(lambda v: f"{v:.0f}" if v == int(v) else str(v), na_action="ignore")
    s = s.astype("object")
    return s.where(s.notna(), np.nan)


def duracion_planeada_dias(df):
    """Duracion planeada del proceso en dias: p_duracion x unidad (dia 1, semana 7, mes 30, anio 365).

    NaN si falta o si la unidad es otra (horas, 'No Especificado'), como en la exploracion.
    """
    unidad = _a_texto(df["p_unidad_de_duracion"]).str.lower().str[0]
    factor = unidad.map({"d": 1, "s": 7, "m": 30, "a": 365}).astype("float64")
    return _a_numero(df["p_duracion"]) * factor


# ---------------------------------------------------------------- funcion unica
def construir_features(df, con_excluidas=False):
    """Devuelve exactamente FEATURES (en ese orden, con dtypes fijos y el mismo indice) a partir de ENTRADAS.

    Recibe el resultado de unir_contrato_proceso: columnas de Contratos y de Procesos con prefijo p_, en texto
    como lo devuelve la API o ya tipadas. Si falta alguna columna de entrada, queda como NaN.
    Con con_excluidas=True agrega al final FEATURES_EXCLUIDAS (solo para la ablacion; el modelo no las usa).
    Es fila a fila (no usa estadisticas de otras filas), idempotente, no lee columnas y* ni eval_* y no hace requests.
    Los faltantes quedan como NaN: la imputacion es del Pipeline.
    """
    x = df.reindex(columns=ENTRADAS)
    firma = _a_dia(x["fecha_de_firma"])
    inicio = _a_dia(x["fecha_de_inicio_del_contrato"])
    publicacion = _a_fecha(x["p_fecha_de_publicacion_del"])

    out = pd.DataFrame(index=df.index)
    for c in ["tipo_de_contrato", "modalidad_de_contratacion", "justificacion_modalidad_de", "origen_de_los_recursos",
              "destino_gasto", "espostconflicto", "es_grupo", "es_pyme", "habilita_pago_adelantado", "documentos_tipo",
              "el_contrato_puede_ser_prorrogado", "tipodocproveedor", "nit_entidad", "sector", "orden", "rama",
              "entidad_centralizada", "departamento", "ciudad", "p_unidad_de_duracion"]:
        out[c] = _a_texto(x[c])

    # UNSPSC: 'V1.' + 8 digitos (segmento = 2, familia = 4); 'UNSPECIFIED' u otro formato -> NaN
    digitos = _a_texto(x["codigo_de_categoria_principal"]).str.extract(r"^V1\.(\d{8})$")[0]
    out["unspsc_segmento"] = digitos.str[:2]
    out["unspsc_familia"] = digitos.str[:4]

    # calendario (H4)
    out["mes_firma"] = firma.dt.month
    out["dia_semana"] = firma.dt.dayofweek
    out["firma_ultimo_trimestre"] = (firma.dt.month >= 10).astype("float64").where(firma.notna())
    dur = duracion_planeada_dias(x)
    fin_planeado = inicio.fillna(firma) + pd.to_timedelta(dur.clip(0, 36500), unit="D")
    out["fin_planeado_cruza_anio"] = (fin_planeado.dt.year > firma.dt.year).astype("float64").where(
        fin_planeado.notna() & firma.notna())

    # contrato
    out["dias_firma_a_inicio"] = (inicio - firma).dt.days.clip(-365, 365)
    out["len_objeto"] = _a_texto(x["objeto_del_contrato"]).fillna("").str.len()
    out["log_pago_adelantado"] = np.log1p(_a_numero(x["valor_de_pago_adelantado"]).clip(lower=0))

    # proceso (H1, H2, H5)
    out["log_precio_base"] = np.log1p(_a_numero(x["p_precio_base"]).clip(lower=0))
    out["p_dur_dias"] = dur
    out["dias_publicacion_a_firma"] = (firma - publicacion).dt.days.clip(-30, 1000)
    for c in ["p_numero_de_lotes", "p_proveedores_invitados", "p_proveedores_con_invitacion",
              "p_proveedores_que_manifestaron", "p_respuestas_al_procedimiento", "p_respuestas_externas",
              "p_conteo_de_respuestas_a_ofertas", "p_proveedores_unicos_con"]:
        out[c] = _a_numero(x[c])

    columnas = FEATURES + (FEATURES_EXCLUIDAS if con_excluidas else [])
    return out[columnas].astype({c: _TIPO[c] for c in columnas})
