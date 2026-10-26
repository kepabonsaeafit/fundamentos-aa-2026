"""J1: re-medicion de las referencias con el marco de evaluacion (src/evaluacion.py).

Mide en test 2024 (contratos maduros, train 2019-2023):
- regla por tasa base y regla por entidad con m=0 y con el m elegido por validacion temporal (elegir_m);
- HGB de la F0 (categorias nativas sobre OrdinalEncoder top-200, random_state=42, resto por defecto) y la variante
  con codigos ordinales (como el dummy, sin early stopping);
- diferencias contra la regla con IC 95 % por bootstrap de entidades, top-k, ROC-AUC dentro de entidad,
  entidades no vistas y tramos de duracion.

Uso (desde final-project/):
  python -m src.medir_j1 --base v0     # comprobacion: debe reproducir 0,714 / 0,277 y ~0,751 / 0,333
  python -m src.medir_j1 --base v1     # referencias reales (base v1_A)
Salida: resultados/j1_<base>_*.csv y resultados/j1_<base>_resumen.md
"""
import argparse
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

RESULTADOS = config.RAIZ / "resultados"
BASES = {"v0": config.DATA_PROCESSED / "base_contratos.parquet",
         "v1": config.DATA_PROCESSED / "v1_A" / "base_contratos.parquet"}


def hgb_f0(nativas):
    """Montaje de la F0 (plan, seccion 0 puntos 7 y 9). nativas=False es el montaje del dummy."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan,
                         max_categories=200)
    prep = ColumnTransformer([("cat", enc, F.FEATURES_CAT)], remainder="passthrough", verbose_feature_names_out=False)
    prep.set_output(transform="pandas")
    hgb = (HistGradientBoostingClassifier(categorical_features=F.FEATURES_CAT, random_state=config.SEED) if nativas
           else HistGradientBoostingClassifier(early_stopping=False, random_state=config.SEED))
    return Pipeline([("prep", prep), ("hgb", hgb)])


def medir(nombre_base, n_boot=200, log=print):
    t0 = time.perf_counter()
    base = pd.read_parquet(BASES[nombre_base])
    X = F.construir_features(base)
    tr, te = E.split_temporal(base, config.ANIO_TEST)
    y_tr, y_te = base.loc[tr, "y"].to_numpy(), base.loc[te, "y"].to_numpy()
    ent_tr, ent_te = base.loc[tr, "nit_entidad"].to_numpy(), base.loc[te, "nit_entidad"].to_numpy()
    m, tabla_m = E.elegir_m(ent_tr, y_tr, base.loc[tr, "anio_firma"].to_numpy(), tabla=True)
    log(f"[{nombre_base}] train {_n(len(tr))} (tasa {y_tr.mean():.4f}) | test {_n(len(te))} (tasa {y_te.mean():.4f}) | m={m:g}")

    scores = {"regla_base": E.regla_tasa_base(y_tr, len(te)),
              "regla_entidad": E.regla_tasa_entidad(ent_tr, y_tr, ent_te, m),
              "regla_entidad_m0": E.regla_tasa_entidad(ent_tr, y_tr, ent_te, 0)}
    for nativas, nombre in ((True, "hgb_f0_nativas"), (False, "hgb_ordinal")):
        pipe = hgb_f0(nativas).fit(X.loc[tr], y_tr)
        scores[nombre] = pipe.predict_proba(X.loc[te])[:, 1]
        log(f"  {nombre} entrenado ({time.perf_counter() - t0:.0f} s)")

    tabla = E.comparar(y_te, scores, ent_te, n_boot=n_boot)
    log(f"  bootstrap global listo ({time.perf_counter() - t0:.0f} s)")
    vista = pd.Series(ent_te).isin(set(ent_tr)).to_numpy()
    tablas_vista = {}
    for etiqueta, mask in (("no_vistas", ~vista), ("vistas", vista)):
        tablas_vista[etiqueta] = E.comparar(y_te[mask], {k: v[mask] for k, v in scores.items()}, ent_te[mask],
                                            n_boot=n_boot).assign(subconjunto=etiqueta)
    tramo = E.tramo_duracion(base.loc[te, "eval_dur_planeada_dias"])
    por_tramo = pd.concat([E.evaluar_por_grupo(y_te, s, tramo).assign(modelo=k) for k, s in scores.items()])
    intra = {k: E.roc_auc_intra_entidad(y_te, s, ent_te, n_boot=n_boot) for k, s in scores.items()
             if k.startswith("hgb")}
    intra20 = {k: E.roc_auc_intra_entidad(y_te, s, ent_te, n_boot=n_boot, min_n=20) for k, s in scores.items()
               if k.startswith("hgb")}

    RESULTADOS.mkdir(exist_ok=True)
    pre = RESULTADOS / f"j1_{nombre_base}"
    tabla.to_csv(f"{pre}_comparacion_2024.csv", index=False)
    pd.concat(tablas_vista.values()).to_csv(f"{pre}_entidades_vistas_2024.csv", index=False)
    por_tramo.to_csv(f"{pre}_por_duracion_2024.csv", index=False)
    tabla_m.to_csv(f"{pre}_elegir_m.csv", index=False)
    resumen = {"base": nombre_base, "n_train": len(tr), "n_test": len(te), "tasa_train": float(y_tr.mean()),
               "tasa_test": float(y_te.mean()), "m_elegido": m, "n_boot": n_boot,
               "entidades_test": int(len(set(ent_te))), "entidades_no_vistas": int(len(set(ent_te[~vista]))),
               "contratos_no_vistos": int((~vista).sum()), "positivos_no_vistos": int(y_te[~vista].sum()),
               "intra_entidad": intra, "intra_entidad_min20": intra20,
               "cumple_criterio": {k: E.cumple_criterio(tabla, k) for k in scores if k.startswith("hgb")},
               "segundos": round(time.perf_counter() - t0)}
    (RESULTADOS / f"j1_{nombre_base}_resumen.json").write_text(json.dumps(resumen, indent=2, default=float), encoding="utf-8")
    (RESULTADOS / f"j1_{nombre_base}_resumen.md").write_text(a_markdown(resumen, tabla, tablas_vista), encoding="utf-8")
    log(f"  listo en {resumen['segundos']} s -> {pre}_*")
    return resumen, tabla, tablas_vista


def _f(x, d=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}".replace(".", ",")


def _n(x):
    return f"{int(x):,}".replace(",", ".")


def _celda(t, modelo, metrica, dif=False):
    fila = t[(t["modelo"] == modelo) & (t["metrica"] == metrica)].iloc[0]
    if dif:
        return f"{_f(fila['dif'])} [{_f(fila['dif_ic_inf'])}, {_f(fila['dif_ic_sup'])}]"
    return f"{_f(fila['valor'])} [{_f(fila['ic_inf'])}, {_f(fila['ic_sup'])}]"


def a_markdown(r, tabla, tablas_vista):
    mets = ["pr_auc", "roc_auc", "p@1%", "p@5%", "p@10%", "r@1%", "r@5%", "r@10%"]
    modelos = list(dict.fromkeys(tabla["modelo"]))
    lin = [f"# J1: referencias con la base {r['base']} (test 2024, train 2019-2023, contratos maduros)", "",
           f"- Train: {_n(r['n_train'])} contratos, tasa {r['tasa_train']:.4f}. Test 2024: {_n(r['n_test'])}, tasa "
           f"{r['tasa_test']:.4f}. m elegido por validacion temporal: {r['m_elegido']:g}.",
           f"- IC 95 % con bootstrap por entidad ({r['n_boot']} repeticiones). [VERIFICADO: salida de src/medir_j1.py]",
           "", "## Global", "", "| Modelo | " + " | ".join(mets) + " |", "|---" * (len(mets) + 1) + "|"]
    for mo in modelos:
        lin.append(f"| {mo} | " + " | ".join(_celda(tabla, mo, me) for me in mets) + " |")
    lin += ["", "Diferencia contra regla_entidad (m elegido):", "", "| Modelo | PR-AUC | ROC-AUC | P@1 % | P@5 % | P@10 % |",
            "|---|---|---|---|---|---|"]
    for mo in modelos:
        lin.append(f"| {mo} | " + " | ".join(_celda(tabla, mo, me, dif=True) for me in ["pr_auc", "roc_auc", "p@1%", "p@5%", "p@10%"]) + " |")
    lin += ["", f"Criterio de aceptacion (IC de la diferencia de PR-AUC excluye 0): {r['cumple_criterio']}", "",
            f"## Entidades no vistas en train ({r['entidades_no_vistas']} entidades, {_n(r['contratos_no_vistos'])} "
            f"contratos, {_n(r['positivos_no_vistos'])} positivos)", "", "| Subconjunto | Modelo | PR-AUC | ROC-AUC | dif PR-AUC | dif ROC-AUC |",
            "|---|---|---|---|---|---|"]
    for sub, t in tablas_vista.items():
        for mo in modelos:
            lin.append(f"| {sub} | {mo} | {_celda(t, mo, 'pr_auc')} | {_celda(t, mo, 'roc_auc')} | "
                       f"{_celda(t, mo, 'pr_auc', True)} | {_celda(t, mo, 'roc_auc', True)} |")
    lin += ["", "## ROC-AUC dentro de entidad (la regla da 0,5 por construccion)", "",
            "| Modelo | Por pares | Por entidad | Entidades | Pares | Por pares (entidades >= 20) |", "|---|---|---|---|---|---|"]
    for mo, v in r["intra_entidad"].items():
        v20 = r["intra_entidad_min20"][mo]
        lin.append(f"| {mo} | {_f(v['pares'])} [{_f(v['pares_ic'][0])}, {_f(v['pares_ic'][1])}] | "
                   f"{_f(v['por_entidad'])} [{_f(v['por_entidad_ic'][0])}, {_f(v['por_entidad_ic'][1])}] | "
                   f"{v['n_entidades']} | {_n(v['n_pares'])} | {_f(v20['pares'])} [{_f(v20['pares_ic'][0])}, {_f(v20['pares_ic'][1])}] |")
    return "\n".join(lin) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=sorted(BASES), default="v1")
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    medir(a.base, a.n_boot)
