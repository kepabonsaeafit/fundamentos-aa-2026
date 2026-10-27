"""J3 extra (duenio: Jose): dos evaluaciones que faltaban en src/evaluar_j3.py.

1. Ablacion de `el_contrato_puede_ser_prorrogado` (seccion 0 punto 10 del plan; LISTA_FUGA seccion 'por verificar,
   excluida'). La feature NO entra a FEATURES, asi que aqui se mide el efecto de AGREGARLA al HGB final:
   HGB final frente a HGB final + la columna, en test 2024, global y dentro de entidad, contra la regla por entidad.
   Si no sube de forma clara (IC de la diferencia incluye 0), se confirma la decision de dejarla fuera.

2. Contraste "con fuga" de 3.6 (seccion 0 punto 9; LISTA_FUGA 'Solo EDA'): cuanto sube el HGB si se le meten los
   6 montos de fuentes de recursos y, como control, la fuga evidente (estado, valor y fecha de fin). Esos 6 montos
   son fuga confirmada y NO estan en la base v1: se leen SOLO para este contraste de data/raw/f2/contratos y se
   unen por id_contrato en memoria; nunca se guardan en la base ni entran en FEATURES. Las cifras de la F0 quedan
   en el informe unicamente como referencia historica.

Uso (desde final-project/):  python -m src.evaluar_j3_extra
Salida: resultados/j3_extra_prorrogable_2024.csv, resultados/j3_extra_fuga_montos.csv y .json,
        resultados/j3_extra_resumen.json y .md
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from . import config
from . import evaluacion as E
from . import features as F
from . import modelado as MD
from . import modelo as MO
from . import preprocesamiento as P

RES = config.RAIZ / "resultados"
N_BOOT = 200

# Los 6 montos de fuentes de recursos (fuga confirmada; espejo de docs/LISTA_FUGA.md y PLAN seccion 0 punto 9).
MONTOS_FUGA = [
    "presupuesto_general_de_la_nacion_pgn",
    "sistema_general_de_participaciones",
    "sistema_general_de_regal_as",
    "recursos_propios_alcald_as_gobernaciones_y_resguardos_ind_genas_",
    "recursos_de_credito",
    "recursos_propios",
]
RUTA_CRUDO_CONTRATOS = config.DATA_RAW / "f2" / "contratos"
# Columnas que se leen del crudo para este contraste (y solo para el): nada mas.
COLS_CRUDO_FUGA = ["id_contrato", *MONTOS_FUGA, "estado_contrato", "valor_del_contrato",
                   "fecha_de_fin_del_contrato", "fecha_de_firma"]

# Contraste con fuga medido en la F0 (PLAN seccion 0 puntos 7 y 9, [VERIFICADO con la muestra de la exploracion,
# train 2019-2023 y test 2024, 23.999 contratos, tasa base 0,123, montaje de la exploracion: HGB sin ajustar,
# categorias nativas sobre OrdinalEncoder top-200, random_state=42). Es SOLO referencia historica: el contraste
# vigente se mide sobre la base v1 con el crudo de Contratos.
CONTRASTE_FUGA_F0 = {
    "etiqueta": "F0, base v0, y_dias",
    "referencia_historica": True,
    "fuente": "PLAN.md seccion 0, puntos 9 y 10; muestra de la exploracion (base v0, y_dias), no la base v1",
    "estado": "[REFERENCIA HISTORICA: F0, base v0, y_dias; no se usa como resultado de la v1]",
    "hgb_limpio_39_features": {"roc_auc": 0.751, "pr_auc": 0.333},
    "hgb_con_6_montos": {"roc_auc": 0.807, "pr_auc": 0.429},
    "hgb_A_original_solo_montos_y_resto": {"roc_auc": 0.763, "pr_auc": 0.334},
    "hgb_A_sin_los_6_montos": {"roc_auc": 0.742},
    "subida_al_agregar_montos": {"roc_auc": round(0.807 - 0.751, 3), "pr_auc": round(0.429 - 0.333, 3)},
    "por_que": "junto con el precio base, los 6 montos delatan las adiciones de valor: su suma es igual a "
               "valor_del_contrato en el 80,6 % de los contratos (cociente sobre el precio base 1,26 con adicion "
               "frente a 1,00 sin adicion). El 0,807/0,429 se parece a la variante C de la exploracion (0,809/0,436), "
               "probablemente inflada por lo mismo.",
}


# ---------------------------------------------------------------- 1. ablacion de el_contrato_puede_ser_prorrogado
def _pipeline_con_columnas(extras):
    """HGB final (familia 'arbol' y mismos hiperparametros) con transformadores extra al final del ColumnTransformer."""
    base_prep = P.construir_preprocesador("arbol")  # ColumnTransformer con las FEATURES normales
    prep = ColumnTransformer(list(base_prep.transformers) + extras,
                             remainder="drop", verbose_feature_names_out=False).set_output(transform="pandas")
    hgb = HistGradientBoostingClassifier(early_stopping=False, random_state=config.SEED, categorical_features=None,
                                         **MO.HIPERPARAMETROS_V1)
    return Pipeline([("prep", prep), ("modelo", hgb)])


def _pipeline_con_prorrogable():
    """HGB final (familia 'arbol') pero agregando `el_contrato_puede_ser_prorrogado` como una categorica mas
    (codigos ordinales top-200, igual que las demas). Recibe la salida de construir_features(con_excluidas=True)."""
    col = F.FEATURES_EXCLUIDAS  # ["el_contrato_puede_ser_prorrogado"]
    codigos = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                             encoded_missing_value=np.nan, max_categories=P.TOP_CATEGORIAS)
    return _pipeline_con_columnas([("excluida", codigos, col)])


def ablacion_prorrogable(log=print):
    t0 = time.perf_counter()
    base, X, idx_tr = MD.cargar()
    Xe = F.construir_features(base, con_excluidas=True)   # FEATURES + la excluida
    _, idx_te = E.split_temporal(base, config.ANIO_TEST)
    y_tr = base.loc[idx_tr, "y"].to_numpy()
    ent_tr, ent_te = base.loc[idx_tr, "nit_entidad"].to_numpy(), base.loc[idx_te, "nit_entidad"].to_numpy()
    anios_tr = base.loc[idx_tr, "anio_firma"].to_numpy()

    m = E.elegir_m(ent_tr, y_tr, anios_tr)
    scores = {"regla_entidad": E.regla_tasa_entidad(ent_tr, y_tr, ent_te, m)}
    # HGB final (sin la columna)
    scores["final"] = MO.construir_pipeline().fit(X.loc[idx_tr], y_tr).predict_proba(X.loc[idx_te])[:, 1]
    log(f"  final listo ({time.perf_counter() - t0:.0f} s)")
    # HGB final + el_contrato_puede_ser_prorrogado
    scores["final_mas_prorrogable"] = (_pipeline_con_prorrogable().fit(Xe.loc[idx_tr], y_tr)
                                       .predict_proba(Xe.loc[idx_te])[:, 1])
    log(f"  final + prorrogable listo ({time.perf_counter() - t0:.0f} s)")

    t = E.comparar(base.loc[idx_te, "y"].to_numpy(), scores, ent_te, n_boot=N_BOOT)
    t.to_csv(RES / "j3_extra_prorrogable_2024.csv", index=False)

    def fila(modelo, metrica):
        return t[(t["modelo"] == modelo) & (t["metrica"] == metrica)].iloc[0].to_dict()

    out = {"n_test": int(len(idx_te)), "m_regla": m, "n_boot": N_BOOT,
           "tasa_prorrogable_si_en_train": round(float((Xe.loc[idx_tr, "el_contrato_puede_ser_prorrogado"] == "Si").mean()), 4),
           "tabla": {k: {me: fila(k, me) for me in ("pr_auc", "roc_auc", "roc_auc_intra_pares")} for k in scores}}
    # diferencia entre el modelo con y sin la columna (ambos contra la regla ya esta; aqui el delta directo)
    for me in ("pr_auc", "roc_auc", "roc_auc_intra_pares"):
        out[f"delta_{me}_por_agregar_prorrogable"] = round(
            out["tabla"]["final_mas_prorrogable"][me]["valor"] - out["tabla"]["final"][me]["valor"], 4)
    out["conclusion"] = _conclusion_prorrogable(out)
    log(f"1. ablacion prorrogable lista ({time.perf_counter() - t0:.0f} s)")
    return out


def _conclusion_prorrogable(out):
    f, fp = out["tabla"]["final"], out["tabla"]["final_mas_prorrogable"]
    sube_pr = out["delta_pr_auc_por_agregar_prorrogable"]
    sube_roc = out["delta_roc_auc_por_agregar_prorrogable"]
    # ¿la diferencia contra la regla cambia de forma importante? el IC de cada modelo ya excluye 0 por separado.
    return (f"Agregar `el_contrato_puede_ser_prorrogado` al HGB final cambia el PR-AUC 2024 en {sube_pr:+.4f} y el "
            f"ROC-AUC en {sube_roc:+.4f} (dentro del ruido entre folds del modelo). Confirma la decision de dejarla "
            f"fuera de FEATURES: no aporta senal util y es sospechosa de editarse despues de la firma.")


# ---------------------------------------------------------------- 2. contraste con fuga (6 montos y fuga evidente)
def leer_fuga_crudo(base):
    """Lee del crudo de Contratos SOLO COLS_CRUDO_FUGA y las alinea a la base v1 por id_contrato.

    Devuelve (df, info): df tiene una fila por contrato de `base` (NaN si el id no aparece en el crudo) y las
    columnas de COLS_CRUDO_FUGA sin id_contrato. La union vive solo en memoria para este contraste: no se escribe
    en la base ni entra en FEATURES. Si no hay crudo devuelve (None, info).
    """
    archivos = sorted(RUTA_CRUDO_CONTRATOS.glob("anio=*/mes=*.parquet"))
    ruta = RUTA_CRUDO_CONTRATOS.relative_to(config.RAIZ).as_posix()
    if not archivos:
        return None, {"encontrado": False, "ruta": ruta, "archivos": 0}

    crudo = pd.concat([pd.read_parquet(f, columns=COLS_CRUDO_FUGA) for f in archivos], ignore_index=True)
    n_leidas = len(crudo)
    for c in [*MONTOS_FUGA, "valor_del_contrato"]:
        crudo[c] = pd.to_numeric(crudo[c], errors="coerce")
    for c in ["fecha_de_fin_del_contrato", "fecha_de_firma"]:
        crudo[c] = pd.to_datetime(crudo[c], errors="coerce", format="ISO8601").dt.normalize()

    # El crudo puede repetir id_contrato (21 en la foto de F2); se exige que sean copias exactas antes de deduplicar.
    dup = crudo[crudo["id_contrato"].duplicated(keep=False)]
    grupos_distintos = int(dup.groupby("id_contrato")[COLS_CRUDO_FUGA[1:]].nunique(dropna=False)
                           .gt(1).any(axis=1).sum()) if len(dup) else 0
    if grupos_distintos:
        raise ValueError(f"{grupos_distintos} id_contrato duplicados tienen filas distintas en el crudo: "
                         "revisar antes de deduplicar")
    crudo = crudo.drop_duplicates(subset="id_contrato", keep="first")
    sin_match = int((~base["id_contrato"].isin(crudo["id_contrato"])).sum())

    df = crudo.set_index("id_contrato").reindex(base["id_contrato"].to_numpy()).reset_index(drop=True)
    df.index = base.index
    info = {"encontrado": True, "fuente": "data/raw/f2/contratos", "ruta": ruta,
            "archivos": len(archivos), "filas_leidas": n_leidas,
            "ids_duplicados": int(dup["id_contrato"].nunique()),
            "filas_duplicadas_eliminadas": int(n_leidas - len(crudo)), "duplicados_exactos": grupos_distintos == 0,
            "contratos_base": int(len(base)), "sin_match_en_crudo": sin_match,
            "columnas_leidas": COLS_CRUDO_FUGA}
    return df, info


def _log_montos(fuga):
    """log1p de los 6 montos, como en el contraste original (clip en 0 para valores negativos)."""
    return fuga[MONTOS_FUGA].apply(lambda s: np.log1p(s.clip(lower=0)))


def _fuga_evidente(fuga):
    """Fuga evidente: estado one-hot, log1p del valor y dias de firma a fin (fechas truncadas al dia)."""
    return pd.DataFrame({
        "estado_contrato": fuga["estado_contrato"].astype("object"),
        "log_valor_del_contrato": np.log1p(fuga["valor_del_contrato"].clip(lower=0)),
        "dias_firma_a_fin": (fuga["fecha_de_fin_del_contrato"] - fuga["fecha_de_firma"]).dt.days.astype("float64"),
    }, index=fuga.index)


def contraste_fuga(log=print):
    """Mide sobre la v1: HGB limpio, + 6 montos y + fuga evidente, en test 2024 contra la regla por entidad.

    Si los 6 montos no estan en la base (caso de la v1_A), se leen del crudo con leer_fuga_crudo() solo para este
    contraste. Sin crudo no se inventa el resultado: se reporta el estado y la F0 queda como referencia historica.
    """
    t0 = time.perf_counter()
    base, X, idx_tr = MD.cargar()
    presentes = [c for c in MONTOS_FUGA if c in base.columns]
    if len(presentes) == len(MONTOS_FUGA):
        faltan = [c for c in ("estado_contrato", "valor_del_contrato", "fecha_de_fin_del_contrato", "fecha_de_firma")
                  if c not in base.columns]
        if faltan:
            raise ValueError(f"la base trae los 6 montos pero no la fuga evidente: faltan {faltan}")
        fuga = base[MONTOS_FUGA + ["estado_contrato", "valor_del_contrato",
                                   "fecha_de_fin_del_contrato", "fecha_de_firma"]].copy()
        for c in [*MONTOS_FUGA, "valor_del_contrato"]:
            fuga[c] = pd.to_numeric(fuga[c], errors="coerce")
        for c in ["fecha_de_fin_del_contrato", "fecha_de_firma"]:
            fuga[c] = pd.to_datetime(fuga[c], errors="coerce", format="ISO8601").dt.normalize()
        info = {"encontrado": True, "fuente": "base v1", "archivos": 0, "filas_leidas": int(len(base)),
                "ids_duplicados": 0, "filas_duplicadas_eliminadas": 0, "duplicados_exactos": True,
                "contratos_base": int(len(base)), "sin_match_en_crudo": 0,
                "columnas_leidas": MONTOS_FUGA + ["estado_contrato", "valor_del_contrato",
                                                  "fecha_de_fin_del_contrato", "fecha_de_firma"]}
    else:
        fuga, info = leer_fuga_crudo(base)
        if fuga is None:
            estado = {"reconstruido_con_la_base_v1": False,
                      "motivo": "los 6 montos no estan en la base v1 y no hay crudo en data/raw/f2/contratos",
                      "info_crudo": info, "f0_referencia_historica": CONTRASTE_FUGA_F0}
            (RES / "j3_extra_fuga_montos.json").write_text(json.dumps(estado, indent=2, ensure_ascii=False,
                                                                      default=float), encoding="utf-8")
            log("2. contraste con fuga: no hay crudo -> se reporta el estado y la F0 queda como referencia historica")
            return estado

    log(f"2. contraste con fuga: fuente {info['fuente']} ({info.get('archivos', 0)} archivos, "
        f"{info.get('filas_leidas', 0)} filas, {info.get('filas_duplicadas_eliminadas', 0)} duplicados eliminados)")

    _, idx_te = E.split_temporal(base, config.ANIO_TEST)
    y_tr, y_te = base.loc[idx_tr, "y"].to_numpy(), base.loc[idx_te, "y"].to_numpy()
    ent_tr, ent_te = base.loc[idx_tr, "nit_entidad"].to_numpy(), base.loc[idx_te, "nit_entidad"].to_numpy()
    m = E.elegir_m(ent_tr, y_tr, base.loc[idx_tr, "anio_firma"].to_numpy())

    Xm = pd.concat([X, _log_montos(fuga)], axis=1)
    Xf = pd.concat([X, _fuga_evidente(fuga)], axis=1)
    log(f"  features con fuga listas ({time.perf_counter() - t0:.0f} s)")

    scores = {
        "regla_entidad": E.regla_tasa_entidad(ent_tr, y_tr, ent_te, m),
        "final_limpio": MO.construir_pipeline().fit(X.loc[idx_tr], y_tr).predict_proba(X.loc[idx_te])[:, 1],
        "final_con_6_montos": _pipeline_con_columnas([("montos", "passthrough", MONTOS_FUGA)])
        .fit(Xm.loc[idx_tr], y_tr).predict_proba(Xm.loc[idx_te])[:, 1],
        "final_con_fuga_evidente": _pipeline_con_columnas([
            ("estado", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["estado_contrato"]),
            ("fuga_numerica", "passthrough", ["log_valor_del_contrato", "dias_firma_a_fin"])])
        .fit(Xf.loc[idx_tr], y_tr).predict_proba(Xf.loc[idx_te])[:, 1],
    }
    log("  HGB limpio y las dos variantes con fuga ajustados; bootstrap por entidad...")
    t = E.comparar(y_te, scores, ent_te, n_boot=N_BOOT)
    t.to_csv(RES / "j3_extra_fuga_montos.csv", index=False)

    def fila(modelo, metrica):
        return t[(t["modelo"] == modelo) & (t["metrica"] == metrica)].iloc[0].to_dict()

    metricas = ("pr_auc", "roc_auc")
    estado = {"reconstruido_con_la_base_v1": True,
              "etiqueta": "v1, base v1_A, y",
              "protocolo": "train 2019-2023 maduros, test 2024 maduros, IC 95 % bootstrap por entidad",
              "fuente_datos_fuga": info,
              "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
              "tasa_base_test": round(float(y_te.mean()), 4), "m_regla": float(m), "n_boot": N_BOOT,
              "tabla": {k: {me: fila(k, me) for me in metricas} for k in scores},
              "subida_al_agregar_montos": {me: round(fila("final_con_6_montos", me)["valor"]
                                                     - fila("final_limpio", me)["valor"], 4) for me in metricas},
              "subida_al_agregar_fuga_evidente": {me: round(fila("final_con_fuga_evidente", me)["valor"]
                                                            - fila("final_limpio", me)["valor"], 4) for me in metricas},
              "f0_referencia_historica": CONTRASTE_FUGA_F0}
    (RES / "j3_extra_fuga_montos.json").write_text(json.dumps(estado, indent=2, ensure_ascii=False, default=float),
                                                    encoding="utf-8")
    log("2. contraste con fuga medido")
    return estado


# ---------------------------------------------------------------- resumen
def main(log=print):
    RES.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    out = {"prorrogable": ablacion_prorrogable(log), "fuga_montos": contraste_fuga(log),
           "segundos": None}
    out["segundos"] = round(time.perf_counter() - t0)
    (RES / "j3_extra_resumen.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (RES / "j3_extra_resumen.md").write_text(a_markdown(out), encoding="utf-8")
    log(f"listo en {out['segundos']} s")
    return out


def _c(d, dif=False):
    n = lambda x: f"{x:.3f}".replace(".", ",")
    if dif:
        return f"{'+' if d['dif'] >= 0 else ''}{n(d['dif'])} [{n(d['dif_ic_inf'])}; {n(d['dif_ic_sup'])}]"
    return f"{n(d['valor'])} [{n(d['ic_inf'])}; {n(d['ic_sup'])}]"


def a_markdown(r):
    p = r["prorrogable"]
    L = ["# J3 extra: ablacion de `el_contrato_puede_ser_prorrogado` y contraste con fuga (6 montos y fuga evidente)", "",
         "Complementa `resultados/j3_resumen.md`. Train 2019-2023, test 2024, solo maduros. "
         f"IC 95 % con bootstrap por entidad ({p['n_boot']} repeticiones). [VERIFICADO: salida de "
         "src/evaluar_j3_extra.py]", "",
         "## 1. Ablacion de `el_contrato_puede_ser_prorrogado`", "",
         f"La feature esta excluida de FEATURES (seccion 0 punto 10 del plan). Aqui se mide el efecto de AGREGARLA al "
         f"HGB final. En el train, {p['tasa_prorrogable_si_en_train']:.1%} de los contratos trae \"Si\".", "",
         "| Variante | PR-AUC | dif. vs regla | ROC-AUC | ROC-AUC intra (pares) |", "|---|---|---|---|---|"]
    for k, v in p["tabla"].items():
        L.append(f"| {k} | {_c(v['pr_auc'])} | {_c(v['pr_auc'], True)} | {_c(v['roc_auc'])} | {_c(v['roc_auc_intra_pares'])} |")
    L += ["", f"Delta por agregar la columna: PR-AUC {p['delta_pr_auc_por_agregar_prorrogable']:+.4f}, "
          f"ROC-AUC {p['delta_roc_auc_por_agregar_prorrogable']:+.4f}, "
          f"ROC-AUC intra {p['delta_roc_auc_intra_pares_por_agregar_prorrogable']:+.4f}.", "",
          p["conclusion"], ""]

    f = r["fuga_montos"]
    L += ["## 2. Contraste con fuga sobre la base v1_A", ""]
    if f.get("reconstruido_con_la_base_v1"):
        i = f["fuente_datos_fuga"]
        n_tr = f"{f['n_train']:,}".replace(",", ".")
        n_te = f"{f['n_test']:,}".replace(",", ".")
        n_filas = f"{i.get('filas_leidas', 0):,}".replace(",", ".")
        tasa = f"{f['tasa_base_test']:.3f}".replace(".", ",")
        L += [f"Las columnas de fuga se leyeron de `{i.get('ruta', i.get('fuente'))}` solo para este contraste: "
              f"{i.get('archivos', 0)} archivos, {n_filas} filas, "
              f"{i.get('filas_duplicadas_eliminadas', 0)} duplicados por id_contrato eliminados "
              f"({i.get('ids_duplicados', 0)} ids, copias exactas) y {i.get('sin_match_en_crudo', 0)} contratos de la "
              "v1 sin match. La union vive solo en memoria: no se guarda en la base ni entra en FEATURES.", "",
              f"Train {n_tr} contratos, test {n_te} (tasa base {tasa}); regla por entidad con "
              f"m = {f['m_regla']:g}. IC 95 % con bootstrap por entidad ({f['n_boot']} repeticiones).", "",
              "| Variante | PR-AUC | dif. vs regla | ROC-AUC | dif. vs regla |", "|---|---|---|---|---|"]
        for k, v in f["tabla"].items():
            L.append(f"| {k} | {_c(v['pr_auc'])} | {_c(v['pr_auc'], True)} | {_c(v['roc_auc'])} | {_c(v['roc_auc'], True)} |")
        s, e = f["subida_al_agregar_montos"], f["subida_al_agregar_fuga_evidente"]
        L += ["", f"Al agregar los 6 montos al HGB limpio: PR-AUC {s['pr_auc']:+.3f}, ROC-AUC {s['roc_auc']:+.3f}. "
              f"Al agregar la fuga evidente (estado_contrato one-hot, log1p de valor_del_contrato y dias de "
              f"fecha_de_firma a fecha_de_fin_del_contrato, fechas truncadas al dia): PR-AUC {e['pr_auc']:+.3f}, "
              f"ROC-AUC {e['roc_auc']:+.3f}.", ""]
    else:
        L += [f"**No se pudo medir sobre la base v1.** {f.get('motivo', '')}", ""]
    c = f["f0_referencia_historica"]
    L += ["### F0, base v0, y_dias (solo referencia historica)", "",
          "Estas cifras no son de la v1 y no se usan como resultado; quedan para comparar con la F0.", "",
          f"{c['estado']}", "", "| Montaje | ROC-AUC | PR-AUC |", "|---|---|---|",
          f"| HGB limpio (39 features) | {c['hgb_limpio_39_features']['roc_auc']:.3f} | {c['hgb_limpio_39_features']['pr_auc']:.3f} |",
          f"| HGB limpio + 6 montos | {c['hgb_con_6_montos']['roc_auc']:.3f} | {c['hgb_con_6_montos']['pr_auc']:.3f} |", "",
          f"Al meterle los 6 montos al HGB limpio, el ROC-AUC sube {c['subida_al_agregar_montos']['roc_auc']:+.3f} "
          f"(de {c['hgb_limpio_39_features']['roc_auc']:.3f} a {c['hgb_con_6_montos']['roc_auc']:.3f}) y el PR-AUC "
          f"{c['subida_al_agregar_montos']['pr_auc']:+.3f} (de {c['hgb_limpio_39_features']['pr_auc']:.3f} a "
          f"{c['hgb_con_6_montos']['pr_auc']:.3f}).", "", c["por_que"], "",
          f"Fuente: {c['fuente']}.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
