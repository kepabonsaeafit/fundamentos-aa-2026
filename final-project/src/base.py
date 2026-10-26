"""Base procesada (contrato 2a): esquema, construccion desde las filas crudas de Contratos y Procesos, y validacion.

Una fila por contrato. La base no contiene columnas de fuga (solo etiquetas y columnas eval_*, que nunca son
features). La base v0 sale de la muestra de la exploracion; la v1 saldra de la descarga completa con el mismo esquema.
Uso para la v0 (desde final-project/):  python -m src.base
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import features as F
from .uniones import elegir_proceso, unir_contrato_proceso

BASE_VERSION = "v0"
FECHA_CORTE_V0 = pd.Timestamp("2026-10-14")   # dia en que se descargo la muestra de la exploracion
DIAS_MADUREZ = 60                             # la prorroga se firma antes del vencimiento; 60 dias de margen [SUPUESTO legal]
DIAS_RESPALDO_SIN_DURACION = 365   # si la duracion planeada es 0 o falta, el plazo es desconocido: madura si la firma tiene mas de 365 dias al corte

LLAVES = ["id_contrato", "proceso_de_compra"]
MOMENTO = ["fecha_de_firma", "anio_firma"]
ENTIDAD = ["nit_entidad", "nombre_entidad"]
ETIQUETAS = ["y", "y_dias", "y_modif"]
EVAL = ["eval_dur_planeada_dias", "eval_fin_planeado", "eval_madura", "eval_en_procesos"]
COLUMNAS_PESOS = ["peso_mes", "N_mes", "n_mes"]   # solo en la base v1 (src/construir_v1.py), al final; no son features

_ENTRADAS_EXTRA = [c for c in F.ENTRADAS if c not in LLAVES + MOMENTO + ENTIDAD]
COLUMNAS_BASE = LLAVES + MOMENTO + ENTIDAD + _ENTRADAS_EXTRA + ETIQUETAS + EVAL

# Tipo de cada columna de entrada: fecha, numero o texto
_FECHAS = ["fecha_de_firma", "fecha_de_inicio_del_contrato", "p_fecha_de_publicacion_del"]
_FECHAS_AL_DIA = ["fecha_de_firma", "fecha_de_inicio_del_contrato"]  # Contratos trae hora desde el 26-oct
_NUMEROS = ["valor_de_pago_adelantado", "p_precio_base", "p_duracion", "p_numero_de_lotes", "p_proveedores_invitados",
            "p_proveedores_con_invitacion", "p_proveedores_que_manifestaron", "p_respuestas_al_procedimiento",
            "p_respuestas_externas", "p_conteo_de_respuestas_a_ofertas", "p_proveedores_unicos_con"]


def tipar_entradas(df):
    """Convierte las columnas de entrada (texto de la API o ya tipadas) a los tipos de la base, con el indice de df."""
    out = pd.DataFrame(index=df.index)
    for c in F.ENTRADAS:
        s = df[c] if c in df else pd.Series(np.nan, index=df.index)
        if c in _FECHAS:
            out[c] = F._a_dia(s) if c in _FECHAS_AL_DIA else F._a_fecha(s)
        elif c in _NUMEROS:
            out[c] = F._a_numero(s).astype("float64")
        else:
            out[c] = F._a_texto(s).astype("str")
    return out


def etiqueta_dias(dias_adicionados):
    """y_dias: el contrato tuvo dias adicionados (> 0). Subestima los positivos (ver docs/01_definicion_problema.md)."""
    return (pd.to_numeric(dias_adicionados, errors="coerce") > 0).astype("int8")


def construir_base(contratos, procesos, fecha_corte, nombres=None):
    """Arma la base procesada a partir de las filas crudas de Contratos y Procesos (texto o tipadas).

    contratos: id_contrato, proceso_de_compra, nombre_entidad, dias_adicionados y las COLS_CONTRATO.
    procesos: filas de Procesos con los nombres de la API (sin prefijo); elegir_proceso deja una por portafolio.
    Solo se conservan los contratos firmados entre ANIO_INICIO y ANIO_DERIVA (2026 queda fuera por censura).
    En la v0, y = y_dias y y_modif = NA (la etiqueta con Modificaciones llega en la v1).
    """
    d = unir_contrato_proceso(contratos, elegir_proceso(procesos))
    x = tipar_entradas(d)
    firma = x["fecha_de_firma"]

    base = pd.DataFrame(index=d.index)
    base["id_contrato"] = F._a_texto(d["id_contrato"]).astype("str")
    base["proceso_de_compra"] = F._a_texto(d["proceso_de_compra"]).astype("str")
    base["fecha_de_firma"] = firma
    base["anio_firma"] = firma.dt.year
    base["nit_entidad"] = x["nit_entidad"]
    base["nombre_entidad"] = F._a_texto(d["nombre_entidad"]).astype("str")
    for c in _ENTRADAS_EXTRA:
        base[c] = x[c]

    base["y_dias"] = etiqueta_dias(d["dias_adicionados"])
    base["y"] = base["y_dias"]
    base["y_modif"] = pd.array([pd.NA] * len(base), dtype="Int8")

    # censura (4.5): el fin planeado sale del proceso (duracion x unidad), nunca de fecha_de_fin_del_contrato
    dur = F.duracion_planeada_dias(x)
    fin = x["fecha_de_inicio_del_contrato"].fillna(firma) + pd.to_timedelta(dur.clip(0, 36500), unit="D")
    base["eval_dur_planeada_dias"] = dur.astype("float32")
    base["eval_fin_planeado"] = fin
    dur_conocida = dur > 0   # 0 y NaN: plazo desconocido (5,2 % con duracion 0 y 0,26 % sin duracion en v0)
    madura_plazo = (fin + pd.Timedelta(days=DIAS_MADUREZ)) <= fecha_corte
    madura_respaldo = (firma + pd.Timedelta(days=DIAS_RESPALDO_SIN_DURACION)) <= fecha_corte
    base["eval_madura"] = madura_plazo.where(dur_conocida, madura_respaldo).fillna(False).astype(bool)
    base["eval_en_procesos"] = d["p_id_del_portafolio"].notna().astype(bool)

    base = base[firma.dt.year.between(config.ANIO_INICIO, config.ANIO_DERIVA)]
    base = base.astype({"anio_firma": "int16"})
    return base[COLUMNAS_BASE].reset_index(drop=True)


def validar_esquema(base):
    """Falla con un mensaje claro si la base no cumple el contrato 2a. Devuelve la misma base."""
    problemas = []
    con_pesos = list(base.columns[len(COLUMNAS_BASE):]) == COLUMNAS_PESOS      # la v1 trae las tres columnas de pesos al final
    esperadas = COLUMNAS_BASE + (COLUMNAS_PESOS if con_pesos else [])
    if list(base.columns) != esperadas:
        faltan = [c for c in esperadas if c not in base]
        sobran = [c for c in base.columns if c not in esperadas]
        problemas.append(f"columnas distintas al esquema (faltan {faltan}, sobran {sobran}) o en otro orden")
    else:
        if con_pesos:
            for c, t in {"peso_mes": "float64", "N_mes": "float64", "n_mes": "int32"}.items():
                if str(base[c].dtype) != t:
                    problemas.append(f"{c} debe ser {t}, es {base[c].dtype}")
            if not ((base["peso_mes"] > 0) & np.isfinite(base["peso_mes"])).all():
                problemas.append("peso_mes debe ser positivo y finito")
        if not base["id_contrato"].is_unique:
            problemas.append("id_contrato no es unico")
        for c in _FECHAS + ["eval_fin_planeado"]:
            if not pd.api.types.is_datetime64_any_dtype(base[c]):
                problemas.append(f"{c} debe ser datetime64, es {base[c].dtype}")
        for c in _NUMEROS:
            if base[c].dtype != "float64":
                problemas.append(f"{c} debe ser float64, es {base[c].dtype}")
        esperado = {"anio_firma": "int16", "y": "int8", "y_dias": "int8", "y_modif": "Int8",
                    "eval_dur_planeada_dias": "float32", "eval_madura": "bool", "eval_en_procesos": "bool"}
        for c, t in esperado.items():
            if str(base[c].dtype) != t:
                problemas.append(f"{c} debe ser {t}, es {base[c].dtype}")
        for c in LLAVES + ENTIDAD + [c for c in _ENTRADAS_EXTRA if c not in _FECHAS + _NUMEROS]:
            if not pd.api.types.is_string_dtype(base[c]):
                problemas.append(f"{c} debe ser texto, es {base[c].dtype}")
        if (base["anio_firma"] != base["fecha_de_firma"].dt.year).any():
            problemas.append("anio_firma no coincide con el anio de fecha_de_firma")
        if not base["y"].isin([0, 1]).all():
            problemas.append("y debe ser 0 o 1")
    fuga = (set(base.columns) & F.COLUMNAS_FUGA) - set(ETIQUETAS) - {"dias_adicionados"}
    if fuga:
        problemas.append(f"columnas de fuga en la base: {sorted(fuga)}")
    if "dias_adicionados" in base:
        problemas.append("dias_adicionados no va en la base (solo y, y_dias, y_modif)")
    if problemas:
        raise ValueError("La base no cumple el esquema 2a:\n  - " + "\n  - ".join(problemas))
    return base


def _sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def construir_base_v0(origen=None, destino=None):
    """Lee la muestra de la exploracion (00_exploracion/data/raw), construye la base v0 y la guarda en data/processed/."""
    origen = Path(origen) if origen else config.RAIZ / "00_exploracion" / "data" / "raw"
    destino = Path(destino) if destino else config.DATA_PROCESSED
    destino.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    cols_c = list(dict.fromkeys(["id_contrato", "proceso_de_compra", "nombre_entidad", "dias_adicionados"] + F.COLS_CONTRATO))
    cols_p = list(dict.fromkeys(["id_del_portafolio", "id_del_proceso", "adjudicado", "fecha_adjudicacion",
                                 "valor_total_adjudicacion"] + [c[2:] for c in F.COLS_PROCESO]))
    contratos = pd.read_parquet(origen / "contratos_muestra.parquet", columns=cols_c)
    procesos = pd.read_parquet(origen / "procesos_muestra.parquet", columns=cols_p)
    base = validar_esquema(construir_base(contratos, procesos, FECHA_CORTE_V0))

    ruta = destino / "base_contratos.parquet"
    base.to_parquet(ruta, index=False)
    por_anio = base.groupby("anio_firma").agg(filas=("y", "size"), tasa_y=("y", "mean"), madura=("eval_madura", "mean"))
    meta = {
        "version": BASE_VERSION,
        "features_version": F.FEATURES_VERSION,
        "fecha_de_corte": str(FECHA_CORTE_V0.date()),
        "origen": "muestra de la exploracion (24.000 contratos por anio de firma), no es la muestra del 10 % de la v1",
        "etiqueta": "y = y_dias = dias_adicionados > 0; y_modif = NA (se llena en la v1 con Modificaciones)",
        "madurez": f"eval_madura: fin planeado (inicio o firma + duracion del proceso) + {DIAS_MADUREZ} dias <= fecha de corte; "
                   "si la duracion planeada es 0 o falta, madura si fecha_de_firma + 365 dias <= fecha de corte",
        "anios": [int(base["anio_firma"].min()), int(base["anio_firma"].max())],
        "filas": int(len(base)),
        "por_anio": {int(a): {"filas": int(r.filas), "tasa_y": round(float(r.tasa_y), 4), "madura": round(float(r.madura), 4)}
                     for a, r in por_anio.iterrows()},
        "columnas": COLUMNAS_BASE,
        "sha256": _sha256(ruta),
        "segundos": round(time.perf_counter() - t0, 1),
    }
    (destino / "base_contratos.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return base, meta


if __name__ == "__main__":
    import psutil
    _base, _meta = construir_base_v0()
    print(f"base v0: {_meta['filas']} filas, {len(_base.columns)} columnas, sha256 {_meta['sha256']}")
    for a, v in _meta["por_anio"].items():
        print(f"  {a}: filas {v['filas']}, tasa y {v['tasa_y']}, maduros {v['madura']}")
    _mi = psutil.Process().memory_info()
    print("RAM pico del proceso (MB):", round(getattr(_mi, "peak_wset", _mi.rss) / 1e6))   # peak_wset solo en Windows
