"""Pesos de la base v1: peso_mes, N_mes y n_mes (PLAN.md 2a; .agents/tasks/f1/muestra_exploracion.md seccion 3.6).

- Modo A (10 % por digito 4, autoponderada): peso_mes = 1,0 constante (decision del usuario). N_mes se llena solo si hay
  conteo exacto; n_mes siempre.
- Modo B (muestra de la exploracion: 2.000 contratos por mes, NO autoponderada): peso_mes = N_mes / n_mes, con
  N_mes = contratos de la poblacion firmados en ese mes y n_mes = filas de la base en ese mes tras deduplicar.
  N_mes sale, en este orden, de (1) el conteo exacto data/raw/f2/conteo_mensual_poblacion.csv (lo escribe A-run al terminar
  su descarga: Contratos por anio y mes de fecha_de_firma, todos los digitos), si existe y su suma por anio cuadra con el total
  anual exacto; (2) la estimacion: contratos con id terminado en 0 por mes (paso4_resultados.json, conteos.contratos_mes_digito)
  escalados al total anual exacto de 00_exploracion/data/raw/meta/poblacion_contratos_por_anio.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

RUTA_CONTEO_EXACTO = config.DATA_RAW / "f2" / "conteo_mensual_poblacion.csv"
RUTA_POBLACION_ANUAL = config.RAIZ / "00_exploracion" / "data" / "raw" / "meta" / "poblacion_contratos_por_anio.csv"
RUTA_PASO4 = config.RAIZ / ".agents" / "tasks" / "f1" / "paso4_resultados.json"
TOL_ANUAL_EXACTO = 0.01      # el conteo exacto (26-oct) puede diferir del total anual del 14-oct en menos de 1 %


def poblacion_anual():
    """poblacion_contratos_por_anio.csv (14-oct, exacta): n y con_dias_adicionados por anio de firma."""
    p = pd.read_csv(RUTA_POBLACION_ANUAL)
    p = p[pd.to_numeric(p["anio"], errors="coerce").notna()].copy()
    p["anio"] = p["anio"].astype(int)
    return p.set_index("anio")


def leer_conteo_exacto(ruta):
    """Lee el CSV de conteo mensual. Acepta (anio, mes, n|conteo|count) o (mes 'AAAA-MM...', n). Devuelve Series 'AAAA-MM' -> n."""
    d = pd.read_csv(ruta)
    d.columns = [str(c).strip().lower() for c in d.columns]
    col_n = next((c for c in ("n", "conteo", "count", "contratos", "total") if c in d), None)
    if col_n is None:
        raise ValueError(f"{ruta}: no encuentro la columna de conteo (n, conteo, count) en {list(d.columns)}")
    if "anio" in d and "mes" in d and pd.to_numeric(d["mes"], errors="coerce").notna().all():
        clave = d["anio"].astype(int).astype(str) + "-" + d["mes"].astype(int).astype(str).str.zfill(2)
    else:
        col_m = next((c for c in ("mes", "anio_mes", "periodo", "mes_firma") if c in d), None)
        if col_m is None:
            raise ValueError(f"{ruta}: no encuentro la columna del mes en {list(d.columns)}")
        clave = d[col_m].astype(str).str[:7]
    s = pd.Series(pd.to_numeric(d[col_n], errors="coerce").to_numpy(), index=clave.to_numpy(), name="N_mes").astype(float)
    if s.isna().any() or s.index.duplicated().any():
        raise ValueError(f"{ruta}: conteos faltantes o meses repetidos")
    return s


def estimar_poblacion_mensual(ruta_paso4=None):
    """N_mes estimado = n0_mes x N_anio / sum(n0_mes del anio), n0 = contratos del digito 0 por mes (muestra_exploracion.md 3.6)."""
    d = json.load(open(ruta_paso4 or RUTA_PASO4, encoding="utf-8"))
    n0 = pd.DataFrame(d["conteos"]["contratos_mes_digito"])
    n0["mes"] = pd.to_datetime(n0["mes"]).dt.strftime("%Y-%m")
    n0 = n0.assign(n0=n0["n"].astype(float)).set_index("mes")[["n0"]]
    n0["anio"] = n0.index.str[:4].astype(int)
    anual = poblacion_anual()["n"]
    n0["N_mes"] = n0["n0"] * n0["anio"].map(anual) / n0.groupby("anio")["n0"].transform("sum")
    return n0["N_mes"]


def poblacion_mensual(modo, conteo_exacto=None):
    """Devuelve (Series N_mes por 'AAAA-MM', fuente, detalle) segun lo disponible. conteo_exacto: ruta o None (usa la de f2)."""
    ruta = Path(conteo_exacto) if conteo_exacto else RUTA_CONTEO_EXACTO
    detalle = {"ruta_conteo_exacto": str(ruta), "existe": ruta.exists()}
    if detalle["existe"]:
        try:
            s = leer_conteo_exacto(ruta)
            anios = s.index.str[:4].astype(int)
            s = s[(anios >= config.ANIO_INICIO) & (anios <= config.ANIO_DERIVA)]
            anual = poblacion_anual()["n"]
            suma = s.groupby(s.index.str[:4].astype(int)).sum()
            desvio = (suma / anual.reindex(suma.index) - 1).abs()
            detalle["desvio_max_vs_total_anual_14oct"] = round(float(desvio.max()), 5)
            if len(s) == 84 and (desvio <= TOL_ANUAL_EXACTO).all():
                return s, "exacto: conteo_mensual_poblacion.csv (group by del servidor)", detalle
            detalle["descartado"] = f"{len(s)} meses y desvio maximo anual {desvio.max():.4f}: no cuadra con el total anual exacto"
        except Exception as e:                       # un archivo ilegible no tumba la construccion: se estima y se avisa
            detalle["descartado"] = f"no se pudo leer: {type(e).__name__}: {e}"
    return estimar_poblacion_mensual(), "estimado: digito 0 por mes escalado al total anual exacto (muestra_exploracion.md 3.6)", detalle


def asignar_pesos(base, modo, conteo_exacto=None):
    """Agrega peso_mes (float64), N_mes (float64) y n_mes (int32) a la base. Devuelve (base, tabla por mes, meta de pesos)."""
    mes = base["fecha_de_firma"].dt.strftime("%Y-%m")
    n_mes = mes.map(mes.value_counts())
    N, fuente, detalle = poblacion_mensual(modo, conteo_exacto)
    N_fila = mes.map(N)
    out = base.copy()
    if modo == "B":
        if N_fila.isna().any():
            falta = sorted(set(mes[N_fila.isna()]))
            raise ValueError(f"modo B: sin N_mes para los meses {falta}")
        out["peso_mes"] = (N_fila / n_mes).astype("float64")
    else:
        out["peso_mes"] = 1.0
        if not fuente.startswith("exacto"):
            N_fila = pd.Series(np.nan, index=base.index)       # en A no se estima: solo se llena con conteo exacto
    out["N_mes"] = N_fila.astype("float64")
    out["n_mes"] = n_mes.astype("int32")

    tabla = pd.DataFrame({"N_mes": N, "n_mes": mes.value_counts()}).dropna(subset=["n_mes"]).sort_index()
    tabla["n_mes"] = tabla["n_mes"].astype(int)
    tabla["peso_mes"] = (tabla["N_mes"] / tabla["n_mes"]) if modo == "B" else 1.0
    tabla["error_rel_esperado_pct"] = (100 * 3 / np.sqrt(tabla["N_mes"])).round(2) if fuente.startswith("estimado") else 0.0
    meta = {"fuente_N_mes": fuente, "detalle": detalle, "modo": modo,
            "formula": "peso_mes = N_mes / n_mes" if modo == "B" else "peso_mes = 1,0 (constante; base autoponderada)"}
    if modo == "B":
        meta["error_esperado"] = error_esperado(tabla, fuente)
    return out, tabla, meta


def error_esperado(tabla, fuente):
    """Error esperado de peso_mes por el error de N_mes. Exacto: 0 (salvo cambios entre el 14 y el 26-oct). Estimado: el digito 0
    es ~10 % de cada mes, asi que n0 ~ Binomial(N, 0,1) y el error relativo de N_mes es ~ sqrt(9 / N_mes) = 3 / sqrt(N_mes),
    mas el sesgo anual del digito 0 que el escalado corrige (-6,8 % en 2019, menos de 2 % desde 2022). Comprobaciones previas
    (muestra_exploracion.md 3.6-3.7): el g observado coincide con el predicho en 78 de 84 meses y con pesos por mes las 7 tasas
    de y_dias quedan dentro del error de muestreo (|z| <= 1,26)."""
    if not fuente.startswith("estimado"):
        return {"tipo": "exacto", "error_relativo_pct": "0 por mes (el conteo es del 26-oct; la muestra, del 14-oct)"}
    e = tabla["error_rel_esperado_pct"]
    return {"tipo": "estimado", "error_relativo_mediano_pct": round(float(e.median()), 2),
            "error_relativo_maximo_pct": round(float(e.max()), 2),
            "mes_de_mayor_error": str(e.idxmax()),
            "regla": "~ 3 / sqrt(N_mes) (binomial del digito 0, p = 0,1)",
            "limite": "no incluye variaciones de la participacion del digito 0 entre meses distintas de la binomial"}


def kish_efectivo(w):
    """Tamano efectivo de muestra de Kish: (sum w)^2 / sum w^2."""
    w = np.asarray(w, dtype="float64")
    return float(w.sum() ** 2 / (w ** 2).sum())
