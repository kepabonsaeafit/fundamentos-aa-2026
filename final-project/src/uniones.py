"""Union de Contratos con Procesos. La usan igual el entrenamiento y la app."""
import numpy as np
import pandas as pd


def elegir_proceso(procesos):
    """Deja una fila por id_del_portafolio (regla de la exploracion).

    Dentro de cada portafolio se elige primero el proceso adjudicado ('Si'), luego el de fecha_adjudicacion
    mas reciente y luego el de sufijo numerico mas alto de id_del_proceso. Un proceso puede tener varias filas
    (una por lote o adjudicacion): se toma la primera y se agregan n_filas y valor_adjudicado_max
    (maximo de valor_total_adjudicacion, que reemplaza a esa columna).
    Recibe las filas de Procesos con los nombres de la API (texto o ya tipadas) y devuelve esas mismas
    columnas, sin prefijo.
    """
    p = procesos.copy()
    for c in ["id_del_portafolio", "id_del_proceso", "adjudicado", "fecha_adjudicacion", "valor_total_adjudicacion"]:
        if c not in p:
            p[c] = np.nan
    p["_adj"] = (p["adjudicado"] == "Si").astype(int)
    p["_fecha_adj"] = pd.to_datetime(p["fecha_adjudicacion"], errors="coerce", format="ISO8601")
    p["_req_num"] = pd.to_numeric(p["id_del_proceso"].astype("string").str.extract(r"(\d+)$")[0], errors="coerce")
    p["_valor"] = pd.to_numeric(p["valor_total_adjudicacion"], errors="coerce")

    orden = p.sort_values(["id_del_portafolio", "_adj", "_fecha_adj", "_req_num"],
                          ascending=[True, False, False, False], na_position="last", kind="mergesort")
    elegido = orden.drop_duplicates("id_del_portafolio")[["id_del_portafolio", "id_del_proceso"]]
    p_sel = p.merge(elegido, on=["id_del_portafolio", "id_del_proceso"], how="inner")

    n_filas = p_sel.groupby("id_del_portafolio").size().rename("n_filas")
    valor_max = p_sel.groupby("id_del_portafolio")["_valor"].max().rename("valor_adjudicado_max")
    salida = (p_sel.drop_duplicates("id_del_portafolio")
              .drop(columns=["_adj", "_fecha_adj", "_req_num", "_valor", "valor_total_adjudicacion"])
              .set_index("id_del_portafolio").join(n_filas).join(valor_max).reset_index())
    assert salida["id_del_portafolio"].is_unique
    return salida


def unir_contrato_proceso(contratos, procesos):
    """Left join Contratos.proceso_de_compra -> Procesos.id_del_portafolio, con prefijo p_ en las de Procesos.

    `procesos` debe venir de elegir_proceso (una fila por portafolio). Falla si la union duplicara contratos.
    Si no hay procesos, devuelve los contratos tal cual (las columnas p_ quedaran como NaN en construir_features).
    """
    p = procesos.add_prefix("p_")
    if "p_id_del_portafolio" not in p:
        return contratos.copy()
    if not p["p_id_del_portafolio"].is_unique:
        raise ValueError("procesos tiene mas de una fila por id_del_portafolio: aplique elegir_proceso antes")
    if "id_contrato" in contratos and not contratos["id_contrato"].is_unique:
        raise ValueError("contratos tiene id_contrato repetidos")
    d = contratos.merge(p, how="left", left_on="proceso_de_compra", right_on="p_id_del_portafolio")
    if len(d) != len(contratos):
        raise ValueError(f"la union cambio el numero de contratos: {len(contratos)} -> {len(d)}")
    d.index = contratos.index
    return d
