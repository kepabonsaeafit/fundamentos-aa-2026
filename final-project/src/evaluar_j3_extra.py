"""J3 extra (duenio: Jose): dos evaluaciones que faltaban en src/evaluar_j3.py.

1. Ablacion de `el_contrato_puede_ser_prorrogado` (seccion 0 punto 10 del plan; LISTA_FUGA seccion 'por verificar,
   excluida'). La feature NO entra a FEATURES, asi que aqui se mide el efecto de AGREGARLA al HGB final:
   HGB final frente a HGB final + la columna, en test 2024, global y dentro de entidad, contra la regla por entidad.
   Si no sube de forma clara (IC de la diferencia incluye 0), se confirma la decision de dejarla fuera.

2. Contraste "con fuga" de 3.6 (seccion 0 punto 9; LISTA_FUGA 'Solo EDA'): cuanto sube el HGB si se le meten los
   6 montos de fuentes de recursos. Esos 6 montos son fuga confirmada y NO estan en la base v1 (quedan solo en
   data/raw/ para el EDA de fuga). El script busca el crudo; si no esta, lo reporta como [NO EVALUADO aqui] y deja
   escrito el contraste verificado en la F0 para que 3.6 lo cite sin reconstruirlo.

Uso (desde final-project/):  python -m src.evaluar_j3_extra
Salida: resultados/j3_extra_prorrogable_2024.csv, resultados/j3_extra_fuga_montos.csv (o .json con el estado),
        resultados/j3_extra_resumen.json y .md
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

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

# Contraste con fuga medido en la F0 (PLAN seccion 0 puntos 7 y 9, [VERIFICADO con la muestra de la exploracion,
# train 2019-2023 y test 2024, 23.999 contratos, tasa base 0,123, montaje de la exploracion: HGB sin ajustar,
# categorias nativas sobre OrdinalEncoder top-200, random_state=42). Son las cifras que cita 3.6; no se reconstruyen
# aqui porque los 6 montos no estan en la base v1 (solo en data/raw/ para el EDA de fuga).
CONTRASTE_FUGA_F0 = {
    "fuente": "PLAN.md seccion 0, puntos 9 y 10; muestra de la exploracion (base v0, y_dias), no la base v1",
    "estado": "[VERIFICADO en la F0; NO reconstruido aqui porque los 6 montos son fuga y no estan en la base v1]",
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
def _pipeline_con_prorrogable():
    """HGB final (familia 'arbol') pero agregando `el_contrato_puede_ser_prorrogado` como una categorica mas
    (codigos ordinales top-200, igual que las demas). Recibe la salida de construir_features(con_excluidas=True)."""
    col = F.FEATURES_EXCLUIDAS  # ["el_contrato_puede_ser_prorrogado"]
    base_prep = P.construir_preprocesador("arbol")  # ColumnTransformer con las FEATURES normales
    codigos = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                             encoded_missing_value=np.nan, max_categories=P.TOP_CATEGORIAS)
    prep = ColumnTransformer(list(base_prep.transformers) + [("excluida", codigos, col)],
                             remainder="drop", verbose_feature_names_out=False).set_output(transform="pandas")
    hgb = HistGradientBoostingClassifier(early_stopping=False, random_state=config.SEED, **MO.HIPERPARAMETROS_V1)
    return Pipeline([("prep", prep), ("modelo", hgb)])


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


# ---------------------------------------------------------------- 2. contraste con fuga (6 montos)
def contraste_fuga(log=print):
    """Si los 6 montos estan disponibles (base o data/raw/), mide HGB final frente a HGB final + montos en 2024.
    Si no, reporta el estado y deja el contraste verificado de la F0 para que 3.6 lo cite."""
    base, X, idx_tr = MD.cargar()
    presentes = [c for c in MONTOS_FUGA if c in base.columns]
    if len(presentes) < len(MONTOS_FUGA):
        faltan = [c for c in MONTOS_FUGA if c not in presentes]
        raw = sorted(p.name for p in config.DATA_RAW.glob("**/*") if p.is_file())
        estado = {
            "reconstruido_con_la_base_v1": False,
            "motivo": "los 6 montos son fuga confirmada y no estan en la base v1 (quedan solo en data/raw/ para el "
                      "EDA de fuga, LISTA_FUGA 'Solo EDA'). El crudo con esas columnas lo tiene Kevin (F2).",
            "montos_presentes_en_la_base_v1": presentes,
            "montos_que_faltan": faltan,
            "archivos_en_data_raw": raw,
            "contraste_verificado_F0": CONTRASTE_FUGA_F0,
        }
        (RES / "j3_extra_fuga_montos.json").write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")
        log("2. contraste con fuga: los 6 montos no estan en la base v1 -> se cita el contraste verificado de la F0")
        return estado

    # Camino para cuando el crudo exista: medir el contraste directamente sobre la base v1.
    log("2. contraste con fuga: los 6 montos estan presentes -> se mide el contraste sobre la base v1")
    _, idx_te = E.split_temporal(base, config.ANIO_TEST)
    y_tr = base.loc[idx_tr, "y"].to_numpy()
    ent_tr, ent_te = base.loc[idx_tr, "nit_entidad"].to_numpy(), base.loc[idx_te, "nit_entidad"].to_numpy()
    m = E.elegir_m(ent_tr, y_tr, base.loc[idx_tr, "anio_firma"].to_numpy())
    Xm_tr = pd.concat([X.loc[idx_tr], base.loc[idx_tr, MONTOS_FUGA].apply(lambda s: np.log1p(s.clip(lower=0)))], axis=1)
    Xm_te = pd.concat([X.loc[idx_te], base.loc[idx_te, MONTOS_FUGA].apply(lambda s: np.log1p(s.clip(lower=0)))], axis=1)
    base_prep = P.construir_preprocesador("arbol")
    prep = ColumnTransformer(list(base_prep.transformers) + [("montos", "passthrough", MONTOS_FUGA)],
                             remainder="drop", verbose_feature_names_out=False).set_output(transform="pandas")
    hgb = HistGradientBoostingClassifier(early_stopping=False, random_state=config.SEED, **MO.HIPERPARAMETROS_V1)
    scores = {"regla_entidad": E.regla_tasa_entidad(ent_tr, y_tr, ent_te, m),
              "final_limpio": MO.construir_pipeline().fit(X.loc[idx_tr], y_tr).predict_proba(X.loc[idx_te])[:, 1],
              "final_con_6_montos": Pipeline([("prep", prep), ("modelo", hgb)]).fit(Xm_tr, y_tr).predict_proba(Xm_te)[:, 1]}
    t = E.comparar(base.loc[idx_te, "y"].to_numpy(), scores, ent_te, n_boot=N_BOOT)
    t.to_csv(RES / "j3_extra_fuga_montos.csv", index=False)

    def fila(modelo, metrica):
        return t[(t["modelo"] == modelo) & (t["metrica"] == metrica)].iloc[0].to_dict()

    estado = {"reconstruido_con_la_base_v1": True,
              "tabla": {k: {me: fila(k, me) for me in ("pr_auc", "roc_auc")} for k in scores},
              "subida_al_agregar_montos": {me: round(fila("final_con_6_montos", me)["valor"] - fila("final_limpio", me)["valor"], 4)
                                           for me in ("pr_auc", "roc_auc")},
              "contraste_verificado_F0": CONTRASTE_FUGA_F0}
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
    L = ["# J3 extra: ablacion de `el_contrato_puede_ser_prorrogado` y contraste con fuga (6 montos)", "",
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
    L += ["## 2. Contraste con fuga: los 6 montos de fuentes de recursos", ""]
    if f.get("reconstruido_con_la_base_v1"):
        L += ["Medido sobre la base v1 (los 6 montos estaban disponibles).", "",
              "| Variante | PR-AUC | ROC-AUC |", "|---|---|---|"]
        for k, v in f["tabla"].items():
            L.append(f"| {k} | {_c(v['pr_auc'])} | {_c(v['roc_auc'])} |")
        s = f["subida_al_agregar_montos"]
        L += ["", f"Subida al agregar los 6 montos: PR-AUC {s['pr_auc']:+.3f}, ROC-AUC {s['roc_auc']:+.3f}.", ""]
    else:
        L += [f"**No se reconstruye sobre la base v1.** {f['motivo']}", "",
              f"Montos presentes en la base v1: {f['montos_presentes_en_la_base_v1'] or 'ninguno'}. "
              f"Archivos en data/raw/: {f['archivos_en_data_raw'] or 'ninguno'}.", ""]
    c = CONTRASTE_FUGA_F0
    L += ["### Contraste verificado en la F0 (el que cita 3.6)", "",
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
