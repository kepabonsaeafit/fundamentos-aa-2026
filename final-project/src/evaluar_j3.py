"""J3: evaluacion final con la base v1 (usa la busqueda de src/modelado.py; el test 2024 se usa una sola vez aqui).

1. Los 5 modelos con sus mejores hiperparametros, entrenados en 2019-2023 (kNN y SVM en la submuestra), contra las
   reglas en test 2024. La eleccion del modelo final se hizo antes, con el PR-AUC de los folds temporales.
2. Categorias nativas frente a codigos ordinales (mismos hiperparametros): en folds temporales y en 2024.
3. Ablacion obligatoria: modelo final sin FEATURES_ABLACION_J3, global y dentro de entidad. Tambien sin las 8
   FEATURES_POR_VERIFICAR.
4. Deriva 2025: (i) mismo modelo 2019-2023; (ii) reentrenado 2019-2024.
5. Sensibilidad de la etiqueta: y frente a y_dias.
6. SHAP global (TreeExplainer, con prueba de aditividad) agregado por feature e hipotesis H1-H7.
7. Entidades vistas / no vistas y tramos de duracion.

Uso (desde final-project/):  python -m src.evaluar_j3
Salida: resultados/j3_*.csv, resultados/j3_resumen.json y resultados/j3_resumen.md
"""
import json
import time

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate

from . import config
from . import evaluacion as E
from . import features as F
from . import modelado as MD
from . import modelo as MO
from . import preprocesamiento as P

RES = config.RAIZ / "resultados"
N_BOOT = 200
N_SHAP = 3000


def _reglas(base, idx_tr, idx_te, etiqueta="y"):
    y_tr = base.loc[idx_tr, etiqueta].to_numpy()
    ent_tr, ent_te = base.loc[idx_tr, "nit_entidad"].to_numpy(), base.loc[idx_te, "nit_entidad"].to_numpy()
    m = E.elegir_m(ent_tr, y_tr, base.loc[idx_tr, "anio_firma"].to_numpy())
    return m, {"regla_base": E.regla_tasa_base(y_tr, len(idx_te)),
               "regla_entidad": E.regla_tasa_entidad(ent_tr, y_tr, ent_te, m)}


def _fila(t, modelo, metrica="pr_auc"):
    return t[(t["modelo"] == modelo) & (t["metrica"] == metrica)].iloc[0].to_dict()


def _comparar(base, idx_te, scores, etiqueta="y", nombre=None):
    t = E.comparar(base.loc[idx_te, etiqueta].to_numpy(), scores, base.loc[idx_te, "nit_entidad"].to_numpy(), n_boot=N_BOOT)
    if nombre:
        t.to_csv(RES / f"j3_{nombre}.csv", index=False)
    return t


def main(log=print):
    t0 = time.perf_counter()
    busq = json.loads((RES / "j3_busqueda_resumen.json").read_text(encoding="utf-8"))
    base, X, idx_tr = MD.cargar()
    _, idx_te = E.split_temporal(base, config.ANIO_TEST)
    y_tr = base.loc[idx_tr, "y"].to_numpy()
    out = {"busqueda": busq}

    # 1. cinco modelos en 2024
    m, scores = _reglas(base, idx_tr, idx_te)
    out["m_regla"] = m
    for nombre, r in busq.items():
        idx = idx_tr
        if MD.MODELOS[nombre][4]:
            rng = np.random.default_rng(config.SEED)
            idx = idx_tr[np.sort(rng.choice(len(idx_tr), MD.N_SUBMUESTRA, replace=False))]
        p = MD.estimador_ajustado(nombre, r["mejores_parametros"]).fit(X.loc[idx], base.loc[idx, "y"])
        scores[nombre] = E.puntaje(p, X.loc[idx_te])
        log(f"  {nombre} en 2024 ({time.perf_counter() - t0:.0f} s)")
    t_modelos = _comparar(base, idx_te, scores, nombre="test2024_modelos")
    out["test2024"] = {k: {me: _fila(t_modelos, k, me) for me in ("pr_auc", "roc_auc", "roc_auc_intra_pares", "p@1%", "p@5%", "p@10%")}
                       for k in scores}
    final = MO.construir_pipeline().fit(X.loc[idx_tr], y_tr)
    s_final = final.predict_proba(X.loc[idx_te])[:, 1]
    assert np.allclose(s_final, scores["hgb"]), "construir_pipeline() no reproduce el HGB de la busqueda"
    log(f"1. modelos listos ({time.perf_counter() - t0:.0f} s)")

    # 2. nativas frente a codigos (mismos hiperparametros)
    hp = {f"modelo__{k}": v for k, v in MO.HIPERPARAMETROS_V1.items()}
    nativo = P.pipeline(HistGradientBoostingClassifier(early_stopping=False, categorical_features="from_dtype",
                                                       random_state=config.SEED), "arbol_nativo").set_params(**hp)
    cv = E.folds_temporales(base.loc[idx_tr, "anio_firma"].to_numpy())
    nat = {}
    for nombre, pipe in (("ordinal", MO.construir_pipeline()), ("nativas", nativo)):
        r = cross_validate(pipe, X.loc[idx_tr], y_tr, cv=cv, scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"})
        nat[nombre] = {"pr_auc_cv": float(r["test_pr_auc"].mean()), "roc_auc_cv": float(r["test_roc_auc"].mean()),
                       "pr_auc_por_fold": r["test_pr_auc"].round(4).tolist()}
    s_nat = nativo.fit(X.loc[idx_tr], y_tr).predict_proba(X.loc[idx_te])[:, 1]
    t_nat = _comparar(base, idx_te, {"regla_entidad": scores["regla_entidad"], "hgb_ordinal": s_final, "hgb_nativas": s_nat})
    for k in ("hgb_ordinal", "hgb_nativas"):
        nat[k.split("_")[1]]["test2024"] = _fila(t_nat, k)
    out["nativas_vs_ordinal"] = nat
    log(f"2. nativas listo ({time.perf_counter() - t0:.0f} s)")

    # 3. ablacion
    abl = {"regla_entidad": scores["regla_entidad"], "final": s_final}
    for nombre, excluir in (("sin_ablacion_j3", F.FEATURES_ABLACION_J3), ("sin_por_verificar", F.FEATURES_POR_VERIFICAR)):
        abl[nombre] = MO.construir_pipeline(excluir).fit(X.loc[idx_tr], y_tr).predict_proba(X.loc[idx_te])[:, 1]
    t_abl = _comparar(base, idx_te, abl, nombre="ablacion_2024")
    out["ablacion"] = {k: {me: _fila(t_abl, k, me) for me in ("pr_auc", "roc_auc", "roc_auc_intra_pares", "roc_auc_intra_entidad")}
                       for k in abl}
    log(f"3. ablacion lista ({time.perf_counter() - t0:.0f} s)")

    # 4. deriva 2025
    der = {}
    tr23, te25 = E.split_temporal(base, config.ANIO_DERIVA, anio_fin_train=config.ANIO_TEST - 1)
    _, sc = _reglas(base, tr23, te25)
    sc["final_2019_2023"] = final.predict_proba(X.loc[te25])[:, 1]
    tr24, _ = E.split_temporal(base, config.ANIO_DERIVA)
    _, sc24 = _reglas(base, tr24, te25)
    sc["regla_entidad_2019_2024"] = sc24["regla_entidad"]
    sc["final_2019_2024"] = MO.construir_pipeline().fit(X.loc[tr24], base.loc[tr24, "y"]).predict_proba(X.loc[te25])[:, 1]
    t_der = _comparar(base, te25, sc, nombre="deriva_2025")
    der = {k: {me: _fila(t_der, k, me) for me in ("pr_auc", "roc_auc", "p@5%")} for k in sc}
    out["deriva_2025"] = {"n": int(len(te25)), "tasa_base": float(base.loc[te25, "y"].mean()), "tabla": der}
    log(f"4. deriva lista ({time.perf_counter() - t0:.0f} s)")

    # 5. sensibilidad de la etiqueta
    _, sd = _reglas(base, idx_tr, idx_te, etiqueta="y_dias")
    sd["final_entrenado_con_y"] = s_final
    sd["final_entrenado_con_y_dias"] = (MO.construir_pipeline().fit(X.loc[idx_tr], base.loc[idx_tr, "y_dias"])
                                        .predict_proba(X.loc[idx_te])[:, 1])
    t_sen = _comparar(base, idx_te, sd, etiqueta="y_dias", nombre="sensibilidad_y_dias_2024")
    out["sensibilidad_y_dias"] = {"tasa_base_y_dias": float(base.loc[idx_te, "y_dias"].mean()),
                                  "tabla": {k: {me: _fila(t_sen, k, me) for me in ("pr_auc", "roc_auc")} for k in sd}}
    log(f"5. sensibilidad lista ({time.perf_counter() - t0:.0f} s)")

    # 6. SHAP global
    muestra = np.random.default_rng(config.SEED).choice(len(idx_te), N_SHAP, replace=False)
    Xt = final[:-1].transform(X.loc[idx_te].iloc[muestra])
    ex = shap.TreeExplainer(final[-1])
    v = ex.shap_values(Xt)
    err = float(np.abs(v.sum(1) + np.ravel(ex.expected_value)[0] - final[-1].decision_function(Xt)).max())
    nombres = [n if n in F.FEATURES else n.rsplit("_", 1)[0] for n in Xt.columns]
    imp = pd.Series(np.abs(v).mean(0), index=nombres).groupby(level=0).sum().sort_values(ascending=False)
    tab = F.tabla_features().set_index("feature")
    shap_t = pd.DataFrame({"media_abs_shap": imp, "hipotesis": tab.loc[imp.index, "hipotesis"],
                           "familia": tab.loc[imp.index, "familia"], "estado_fuga": tab.loc[imp.index, "estado_fuga"]})
    shap_t.to_csv(RES / "j3_shap_global_2024.csv")
    out["shap"] = {"n": N_SHAP, "error_aditividad_max": err, "top10": shap_t.head(10)["media_abs_shap"].round(4).to_dict(),
                   "por_hipotesis": shap_t.groupby("hipotesis")["media_abs_shap"].sum().sort_values(ascending=False).round(4).to_dict()}
    log(f"6. SHAP listo, error de aditividad {err:.2e} ({time.perf_counter() - t0:.0f} s)")

    # 7. grupos
    ent_tr = set(base.loc[idx_tr, "nit_entidad"])
    vista = np.where(base.loc[idx_te, "nit_entidad"].isin(ent_tr), "vista", "no vista")
    tramo = E.tramo_duracion(base.loc[idx_te, "eval_dur_planeada_dias"])
    y_te = base.loc[idx_te, "y"].to_numpy()
    grupos = pd.concat([E.evaluar_por_grupo(y_te, s, g).assign(modelo=k, tipo=t)
                        for k, s in (("final", s_final), ("regla_entidad", scores["regla_entidad"]))
                        for t, g in (("entidad", vista), ("duracion", tramo))])
    grupos.to_csv(RES / "j3_por_grupo_2024.csv", index=False)
    noviste = vista == "no vista"
    t_nv = E.comparar(y_te[noviste], {"regla_entidad": scores["regla_entidad"][noviste], "final": s_final[noviste]},
                      base.loc[idx_te, "nit_entidad"].to_numpy()[noviste], n_boot=N_BOOT)
    out["no_vistas"] = {"n": int(noviste.sum()), "positivos": int(y_te[noviste].sum()),
                        "final": {me: _fila(t_nv, "final", me) for me in ("pr_auc", "roc_auc")}}

    out["criterio_aceptacion_final"] = E.cumple_criterio(t_modelos, "hgb")
    out["segundos"] = round(time.perf_counter() - t0)
    (RES / "j3_resumen.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (RES / "j3_resumen.md").write_text(a_markdown(out), encoding="utf-8")
    log(f"listo en {out['segundos']} s")
    return out, final, s_final


def _c(d, dif=False):
    n = lambda x: f"{x:.3f}".replace(".", ",")
    if dif:
        return f"{'+' if d['dif'] >= 0 else ''}{n(d['dif'])} [{n(d['dif_ic_inf'])}; {n(d['dif_ic_sup'])}]"
    return f"{n(d['valor'])} [{n(d['ic_inf'])}; {n(d['ic_sup'])}]"


def a_markdown(r):
    """Resumen de J3 para el notebook (todas las cifras salen de j3_resumen.json)."""
    L = ["# J3: modelado y seleccion con la base v1_A", "",
         "Train 2019-2023, test 2024 y deriva 2025, solo contratos maduros. IC 95 % con bootstrap por entidad "
         f"({N_BOOT} repeticiones). Regla por entidad con m = {r['m_regla']:g} (elegido con folds temporales). "
         "[VERIFICADO: salida de src/modelado.py y src/evaluar_j3.py]", "",
         "## Busqueda (folds temporales 2021-2023, PR-AUC medio)", "",
         "| Modelo | n train | n_iter | Segundos | PR-AUC cv | ROC-AUC cv | Mejores parametros |", "|---|---|---|---|---|---|---|"]
    for k, b in r["busqueda"].items():
        L.append(f"| {k} | {b['n_train']:,} | ".replace(",", ".") + f"{b['n_iter']} | {b['segundos']} | {b['pr_auc_cv']:.4f} ± "
                 f"{b['pr_auc_cv_std']:.4f} | {b['roc_auc_cv']:.4f} | `{b['mejores_parametros']}` |")
    L += ["", "## Test 2024 (una sola corrida)", "", "| Modelo | PR-AUC | dif. PR-AUC vs regla | ROC-AUC | ROC-AUC intra-entidad | P@1 % | P@5 % | P@10 % |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in r["test2024"].items():
        L.append(f"| {k} | {_c(v['pr_auc'])} | {_c(v['pr_auc'], True)} | {_c(v['roc_auc'])} | {_c(v['roc_auc_intra_pares'])} | "
                 + " | ".join(f"{v[p]['valor']:.3f}".replace(".", ",") for p in ("p@1%", "p@5%", "p@10%")) + " |")
    L += ["", f"Criterio de aceptacion del HGB final (IC de la dif. de PR-AUC excluye 0): **{r['criterio_aceptacion_final']}**", "",
          "## Categorias nativas frente a codigos ordinales (mismos hiperparametros)", "", "| Variante | PR-AUC cv | ROC-AUC cv | PR-AUC por fold | PR-AUC 2024 |", "|---|---|---|---|---|"]
    for k, v in r["nativas_vs_ordinal"].items():
        L.append(f"| {k} | {v['pr_auc_cv']:.4f} | {v['roc_auc_cv']:.4f} | {v['pr_auc_por_fold']} | {_c(v['test2024'])} |")
    L += ["", "## Ablacion (test 2024)", "", "| Variante | PR-AUC | dif. vs regla | ROC-AUC | ROC-AUC intra (pares) | dif. intra vs regla |", "|---|---|---|---|---|---|"]
    for k, v in r["ablacion"].items():
        L.append(f"| {k} | {_c(v['pr_auc'])} | {_c(v['pr_auc'], True)} | {_c(v['roc_auc'])} | {_c(v['roc_auc_intra_pares'])} | {_c(v['roc_auc_intra_pares'], True)} |")
    d = r["deriva_2025"]
    L += ["", f"## Deriva 2025 (n = {d['n']:,}, tasa base {d['tasa_base']:.4f})".replace(",", "."), "",
          "| Puntaje | PR-AUC | dif. vs regla 2019-2023 | ROC-AUC |", "|---|---|---|---|"]
    for k, v in d["tabla"].items():
        L.append(f"| {k} | {_c(v['pr_auc'])} | {_c(v['pr_auc'], True)} | {_c(v['roc_auc'])} |")
    s = r["sensibilidad_y_dias"]
    L += ["", f"## Sensibilidad: evaluado con y_dias en 2024 (tasa base {s['tasa_base_y_dias']:.4f})", "",
          "| Puntaje | PR-AUC | dif. vs regla (y_dias) | ROC-AUC |", "|---|---|---|---|"]
    for k, v in s["tabla"].items():
        L.append(f"| {k} | {_c(v['pr_auc'])} | {_c(v['pr_auc'], True)} | {_c(v['roc_auc'])} |")
    nv = r["no_vistas"]
    L += ["", f"## Entidades no vistas en train ({nv['n']:,} contratos, {nv['positivos']} positivos)".replace(",", "."), "",
          f"HGB final: PR-AUC {_c(nv['final']['pr_auc'])}, dif. vs regla {_c(nv['final']['pr_auc'], True)}; "
          f"ROC-AUC {_c(nv['final']['roc_auc'])}, dif. {_c(nv['final']['roc_auc'], True)}.", "",
          f"## SHAP global (TreeExplainer, {r['shap']['n']} contratos de 2024; error de aditividad maximo "
          f"{r['shap']['error_aditividad_max']:.1e})", "", "| Feature | media |SHAP| (log-odds) |", "|---|---|"]
    L += [f"| {k} | {v:.4f} |" for k, v in r["shap"]["top10"].items()]
    L += ["", "Por hipotesis:", "", "| Hipotesis | suma media |SHAP| |", "|---|---|"]
    L += [f"| {k} | {v:.4f} |" for k, v in r["shap"]["por_hipotesis"].items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
