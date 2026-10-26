"""Etiqueta de la base v1: ampliacion del plazo (PLAN.md 4.3, docs/LISTA_FUGA.md seccion 3).

y = 1 si dias_adicionados > 0 (y_dias) o existe una modificacion con estado 'Publicado', dias_extendidos > 0 y
fecha_de_aprobacion > fecha_de_firma + 1 dia (y_modif). Las suspensiones y los ajustes de fecha cuentan como positivos.

Las unidades de Modificaciones de F2 ya traen solo estado 'Publicado' con dias_extendidos > 0 y fecha_version <= corte
(filtro de servidor); aqui se vuelven a aplicar los dos primeros (no cuestan nada) y el corte solo se cuenta, no se
aplica, salvo que se pida (la zona horaria de fecha_version la fija B2; ver docs/BASE_V1.md).
Todo es fila a fila por contrato: no usa estadisticas de otras filas ni duraci_n_del_contrato.
"""
import numpy as np
import pandas as pd

from . import features as F

# Columnas minimas del paso 3 de la F1 (PLAN.md 4.2); identificador_modificacion, version_anterior_del_contrato y
# fecha_version sirven para deduplicar y ordenar.
COLS_MODIF = ["id_contrato", "identificador_modificacion", "version_anterior_del_contrato", "fecha_version",
              "estado_modificacion", "dias_extendidos", "fecha_de_aprobacion", "fecha_fin_contrato"]
ESTADO_EFECTIVO = "Publicado"
MARGEN_DIAS = 1      # fecha_de_aprobacion debe superar fecha_de_firma + 1 dia (desfase de zona horaria, paso 2 de la F1)


def preparar_modificaciones(modif, corte=None, aplicar_corte=False):
    """Deduplica filas exactas, filtra y tipa Modificaciones. Devuelve (tabla tipada, info con los conteos).

    modif: filas de Modificaciones como las baja F2 (texto), con COLS_MODIF.
    corte: Timestamp (sin zona) para contar las filas con fecha_version posterior; solo filtra si aplicar_corte.
    """
    info = {"filas_leidas": int(len(modif))}
    m = modif.reindex(columns=COLS_MODIF)
    antes = len(m)
    m = m.drop_duplicates()
    info["duplicados_exactos"] = int(antes - len(m))
    info["pct_duplicados_exactos"] = round(100 * info["duplicados_exactos"] / antes, 2) if antes else 0.0

    m = m.assign(
        dias=F._a_numero(m["dias_extendidos"]),
        aprobacion=F._a_fecha(m["fecha_de_aprobacion"]),
        version=F._a_fecha(m["fecha_version"]),
    )
    efectiva = (m["estado_modificacion"] == ESTADO_EFECTIVO) & (m["dias"] > 0)
    info["filas_no_efectivas"] = int((~efectiva).sum())
    m = m[efectiva]
    info["filas_efectivas"] = int(len(m))
    info["efectivas_sin_fecha_de_aprobacion"] = int(m["aprobacion"].isna().sum())

    if corte is not None:
        posteriores = m["version"] > pd.Timestamp(corte)
        info["fecha_version_posterior_al_corte"] = int(posteriores.sum())
        if aplicar_corte:
            m = m[~posteriores]
            info["corte_aplicado"] = True
    info.setdefault("corte_aplicado", False)
    return m.reset_index(drop=True), info


def etiquetar(base, modif_prep):
    """Devuelve (base con y, y_dias, y_modif, aportes). `base` trae id_contrato, fecha_de_firma y y_dias (de base.py).

    y_modif es 0 o 1 para todos los contratos (F2 consulto Modificaciones de todos); Int8 como pide el esquema 2a.
    """
    firma = base.set_index("id_contrato")["fecha_de_firma"]
    m = modif_prep.assign(firma=modif_prep["id_contrato"].map(firma))
    en_base = m["firma"].notna()
    cumple = en_base & (m["aprobacion"] > m["firma"] + pd.Timedelta(days=MARGEN_DIAS))
    ids_modif = set(m.loc[cumple, "id_contrato"])
    ids_sin_margen = set(m.loc[en_base, "id_contrato"])

    out = base.copy()
    out["y_modif"] = out["id_contrato"].isin(ids_modif).astype("Int8")
    out["y"] = ((out["y_dias"] == 1) | (out["y_modif"] == 1)).astype("int8")
    y_dias, y_modif = out["y_dias"].astype(bool), out["y_modif"].astype(bool)
    aportes = {
        "contratos": int(len(out)),
        "positivos_y": int(out["y"].sum()),
        "solo_y_dias": int((y_dias & ~y_modif).sum()),
        "solo_y_modif": int((~y_dias & y_modif).sum()),
        "ambas": int((y_dias & y_modif).sum()),
        "positivos_y_dias": int(y_dias.sum()),
        "positivos_y_modif": int(y_modif.sum()),
        "contratos_con_extension_publicada": int(len(ids_sin_margen)),
        "con_extension_pero_no_cumple_margen": int(len(ids_sin_margen - ids_modif)),
        "modificaciones_de_contratos_fuera_de_la_base": int((~en_base).sum()),
    }
    return out, aportes
