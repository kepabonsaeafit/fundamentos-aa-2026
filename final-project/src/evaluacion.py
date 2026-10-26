"""Marco de evaluacion (contrato 2c, duenio: Jose).

Split temporal con contratos maduros, folds temporales para la busqueda, las dos reglas simples (tasa base y tasa
historica por entidad), metricas con top-k, ROC-AUC dentro de entidad y comparacion contra la regla con IC 95 % por
bootstrap POR ENTIDAD (los contratos de una misma entidad estan correlacionados; remuestrear contratos da intervalos
optimistas, plan seccion 0 punto 2).

Criterio de aceptacion (plan 2c): en test 2024, el IC 95 % de la diferencia de PR-AUC contra la regla por entidad
excluye el cero.

La base v1_A es autoponderada (peso_mes = 1), asi que las metricas no se ponderan.
"""
import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config
from . import features as F

FRACCIONES_K = (0.01, 0.05, 0.10)
M_CANDIDATOS = (0, 1, 2, 5, 10, 20, 50, 100)
TOLERANCIA_ROC_M = 0.005   # elegir_m: entre las m a menos de esto del mejor ROC-AUC, gana la de mejor precision top-k


# ---------------------------------------------------------------- split
def split_temporal(base, anio_test, anio_inicio=config.ANIO_INICIO, anio_fin_train=None):
    """(idx_train, idx_test): train = [anio_inicio, anio_fin_train] (por defecto hasta anio_test - 1), test = anio_test.

    Ambos solo con contratos maduros (eval_madura). Nunca es aleatorio (regla 3).
    anio_fin_train permite la deriva (i): modelo 2019-2023 evaluado en 2025.
    """
    fin = anio_test - 1 if anio_fin_train is None else anio_fin_train
    if fin >= anio_test:
        raise ValueError("el train debe terminar antes del anio de test")
    madura = base["eval_madura"].astype(bool)
    anio = base["anio_firma"]
    idx_train = base.index[madura & (anio >= anio_inicio) & (anio <= fin)]
    idx_test = base.index[madura & (anio == anio_test)]
    return idx_train, idx_test


def folds_temporales(anios, anios_val=(2021, 2022, 2023)):
    """Ventana creciente: para cada anio de validacion v, train = anios < v y val = anios == v.

    Devuelve posiciones (no etiquetas del indice), asi que sirve directo como cv= de scikit-learn.
    """
    anios = np.asarray(anios)
    folds = []
    for v in anios_val:
        tr, va = np.flatnonzero(anios < v), np.flatnonzero(anios == v)
        if len(tr) and len(va):
            folds.append((tr, va))
    if not folds:
        raise ValueError("ningun fold temporal tiene train y validacion")
    return folds


# ---------------------------------------------------------------- reglas simples (regla 5)
def regla_tasa_base(y_train, n):
    """Puntaje constante = tasa de positivos del train."""
    return np.full(n, float(np.mean(y_train)))


def regla_tasa_entidad(ent_train, y_train, ent_test, m):
    """Tasa historica suavizada por entidad: (pos + m * p0) / (n + m). Entidades no vistas -> p0."""
    y_train = np.asarray(y_train, dtype="float64")
    p0 = float(y_train.mean())
    g = pd.DataFrame({"e": np.asarray(ent_train), "y": y_train}).groupby("e")["y"].agg(["sum", "count"])
    tasa = (g["sum"] + m * p0) / (g["count"] + m)
    return pd.Series(np.asarray(ent_test)).map(tasa).fillna(p0).to_numpy(dtype="float64")


# ---------------------------------------------------------------- metricas
def _precision_top_k(y, score, k):
    """Precision esperada en los k primeros con desempate aleatorio (la regla por entidad tiene muchos empates)."""
    orden = np.sort(score)[::-1]
    umbral = orden[k - 1]
    arriba = score > umbral
    en_umbral = score == umbral
    faltan = k - int(arriba.sum())
    return (y[arriba].sum() + faltan * y[en_umbral].mean()) / k


def metricas(y_true, score, fracciones_k=FRACCIONES_K):
    """n, positivos, tasa base, PR-AUC, ROC-AUC, PR-AUC / tasa base y precision y recall en el top-k."""
    y = np.asarray(y_true, dtype="float64")
    s = np.asarray(score, dtype="float64")
    n, pos = len(y), float(y.sum())
    base = pos / n if n else np.nan
    dos_clases = 0 < pos < n
    out = {"n": n, "positivos": int(pos), "tasa_base": base,
           "pr_auc": average_precision_score(y, s) if dos_clases else np.nan,
           "roc_auc": roc_auc_score(y, s) if dos_clases else np.nan}
    out["pr_auc_sobre_base"] = out["pr_auc"] / base if dos_clases else np.nan
    for f in fracciones_k:
        k = max(1, math.ceil(f * n))
        p = _precision_top_k(y, s, k) if n else np.nan
        out[f"p@{f:.0%}"] = p
        out[f"r@{f:.0%}"] = p * k / pos if pos else np.nan
    return out


def _auc_por_entidad(y_true, score, entidad):
    """AUC de Mann-Whitney dentro de cada entidad con ambas clases (empates = 0,5). DataFrame: auc, pares."""
    d = pd.DataFrame({"y": np.asarray(y_true, dtype="float64"), "s": np.asarray(score, dtype="float64"),
                      "e": np.asarray(entidad)})
    d["r"] = d.groupby("e")["s"].rank(method="average")
    g = d.groupby("e").agg(n=("y", "size"), pos=("y", "sum"))
    g["suma_r_pos"] = d[d["y"] == 1].groupby("e")["r"].sum()
    g = g[(g["pos"] > 0) & (g["pos"] < g["n"])].copy()
    neg = g["n"] - g["pos"]
    g["pares"] = g["pos"] * neg
    g["auc"] = (g["suma_r_pos"] - g["pos"] * (g["pos"] + 1) / 2) / g["pares"]
    return g[["auc", "pares", "n"]]


def _resumen_intra(g):
    if g.empty:
        return np.nan, np.nan
    return float(np.average(g["auc"], weights=g["pares"])), float(g["auc"].mean())


def roc_auc_intra_entidad(y_true, score, entidad, n_boot=0, seed=config.SEED, min_n=None):
    """ROC-AUC dentro de entidad: P(score prorrogado > score no prorrogado | misma entidad).

    Devuelve la version ponderada por pares y el promedio por entidad (solo entidades con ambas clases).
    Con n_boot > 0 agrega IC 95 % con bootstrap de entidades. La regla por entidad da 0,5 por construccion.
    min_n filtra entidades con al menos min_n contratos (sensibilidad).
    """
    g = _auc_por_entidad(y_true, score, entidad)
    if min_n is not None:
        g = g[g["n"] >= min_n]
    pares, por_entidad = _resumen_intra(g)
    out = {"pares": pares, "por_entidad": por_entidad, "n_entidades": int(len(g)), "n_pares": int(g["pares"].sum())}
    if n_boot and len(g):
        rng = np.random.default_rng(seed)
        auc, w = g["auc"].to_numpy(), g["pares"].to_numpy()
        bp, be = [], []
        for _ in range(n_boot):
            i = rng.integers(0, len(g), len(g))
            bp.append(np.average(auc[i], weights=w[i]))
            be.append(auc[i].mean())
        out["pares_ic"] = tuple(np.percentile(bp, [2.5, 97.5]))
        out["por_entidad_ic"] = tuple(np.percentile(be, [2.5, 97.5]))
    return out


# ---------------------------------------------------------------- elegir m (validacion temporal dentro del train)
def elegir_m(ent, y, anios, ms=M_CANDIDATOS, anios_val=(2021, 2022, 2023), fracciones_k=FRACCIONES_K,
             tolerancia=TOLERANCIA_ROC_M, tabla=False):
    """Elige el suavizado m de la regla por entidad con los folds temporales del train (nunca con el test).

    Regla: entre las m cuyo ROC-AUC medio queda a menos de `tolerancia` del mejor, gana la de mayor precision media
    en el top-k (promedio de P@1 %, 5 % y 10 %). Con m=0 el top 1 % queda dominado por entidades con 1 de 1
    positivos (plan, seccion 0 punto 11), por eso no basta el ROC-AUC.
    Con tabla=True devuelve (m, DataFrame con las metricas por m).
    """
    ent, y, anios = np.asarray(ent), np.asarray(y), np.asarray(anios)
    filas = []
    for m in ms:
        por_fold = []
        for tr, va in folds_temporales(anios, anios_val):
            s = regla_tasa_entidad(ent[tr], y[tr], ent[va], m)
            por_fold.append(metricas(y[va], s, fracciones_k))
        d = pd.DataFrame(por_fold)
        fila = {"m": m, "roc_auc": d["roc_auc"].mean(), "pr_auc": d["pr_auc"].mean()}
        for f in fracciones_k:
            fila[f"p@{f:.0%}"] = d[f"p@{f:.0%}"].mean()
        fila["p_topk_media"] = np.mean([fila[f"p@{f:.0%}"] for f in fracciones_k])
        filas.append(fila)
    t = pd.DataFrame(filas)
    candidatas = t[t["roc_auc"] >= t["roc_auc"].max() - tolerancia]
    m = float(candidatas.sort_values(["p_topk_media", "roc_auc"], ascending=False).iloc[0]["m"])
    t["elegida"] = t["m"] == m
    return (m, t) if tabla else m


# ---------------------------------------------------------------- comparacion con bootstrap por entidad
def _grupos(entidad):
    codigos, _ = pd.factorize(pd.Series(np.asarray(entidad)), use_na_sentinel=False)
    orden = np.argsort(codigos, kind="stable")
    limites = np.r_[0, np.cumsum(np.bincount(codigos))]
    return [orden[limites[i]:limites[i + 1]] for i in range(len(limites) - 1)]


def _metricas_boot(y, s, ent, fracciones_k):
    m = metricas(y, s, fracciones_k)
    m["roc_auc_intra_pares"], m["roc_auc_intra_entidad"] = _resumen_intra(_auc_por_entidad(y, s, ent))
    return m


METRICAS_COMPARAR = ["pr_auc", "roc_auc", "pr_auc_sobre_base", "roc_auc_intra_pares", "roc_auc_intra_entidad"]


def comparar(y_true, scores, entidad, referencia="regla_entidad", n_boot=200, seed=config.SEED,
             fracciones_k=FRACCIONES_K):
    """Metricas de cada puntaje y diferencia contra `referencia`, con IC 95 % por bootstrap POR ENTIDAD.

    scores: dict nombre -> puntaje (mismo orden que y_true). Devuelve un DataFrame largo:
    modelo, metrica, valor, ic_inf, ic_sup, dif, dif_ic_inf, dif_ic_sup.
    """
    y = np.asarray(y_true, dtype="float64")
    ent = np.asarray(entidad)
    scores = {k: np.asarray(v, dtype="float64") for k, v in scores.items()}
    if referencia not in scores:
        raise KeyError(f"la referencia {referencia!r} no esta en scores")
    nombres_m = METRICAS_COMPARAR + [f"{t}@{f:.0%}" for f in fracciones_k for t in ("p", "r")]
    punto = {k: _metricas_boot(y, s, ent, fracciones_k) for k, s in scores.items()}

    grupos = _grupos(ent)
    rng = np.random.default_rng(seed)
    boot = {k: {m: [] for m in nombres_m} for k in scores}
    for _ in range(n_boot):
        elegidos = rng.integers(0, len(grupos), len(grupos))
        idx = np.concatenate([grupos[i] for i in elegidos])
        # etiqueta de entidad por repeticion: una entidad elegida dos veces cuenta como dos entidades
        ent_b = np.concatenate([np.full(len(grupos[i]), j) for j, i in enumerate(elegidos)])
        for k, s in scores.items():
            mb = _metricas_boot(y[idx], s[idx], ent_b, fracciones_k)
            for m in nombres_m:
                boot[k][m].append(mb[m])

    filas = []
    for k in scores:
        for m in nombres_m:
            v = np.asarray(boot[k][m], dtype="float64")
            r = np.asarray(boot[referencia][m], dtype="float64")
            dif = v - r
            ic = np.nanpercentile(v, [2.5, 97.5]) if n_boot else (np.nan, np.nan)
            dic = np.nanpercentile(dif, [2.5, 97.5]) if n_boot else (np.nan, np.nan)
            filas.append({"modelo": k, "metrica": m, "valor": punto[k][m], "ic_inf": ic[0], "ic_sup": ic[1],
                          "dif": punto[k][m] - punto[referencia][m], "dif_ic_inf": dic[0], "dif_ic_sup": dic[1]})
    t = pd.DataFrame(filas)
    t.attrs.update({"n": len(y), "tasa_base": float(y.mean()), "n_entidades": len(grupos), "n_boot": n_boot,
                    "referencia": referencia})
    return t


def cumple_criterio(tabla, modelo, metrica="pr_auc"):
    """Criterio de aceptacion del plan (2c): el IC 95 % de la diferencia de PR-AUC contra la regla excluye el cero."""
    fila = tabla[(tabla["modelo"] == modelo) & (tabla["metrica"] == metrica)].iloc[0]
    return bool(fila["dif_ic_inf"] > 0)


def tabla_ancha(tabla, metricas_=("pr_auc", "roc_auc", "roc_auc_intra_pares", "p@1%", "p@5%", "p@10%"), dec=3):
    """Tabla corta para el notebook: 'valor [ic]' por modelo y metrica, y la diferencia contra la referencia."""
    t = tabla[tabla["metrica"].isin(metricas_)].copy()
    fmt = lambda a, b, c: f"{a:.{dec}f} [{b:.{dec}f}, {c:.{dec}f}]"
    t["valor_ic"] = [fmt(a, b, c) for a, b, c in zip(t["valor"], t["ic_inf"], t["ic_sup"])]
    t["dif_ic"] = [fmt(a, b, c) if np.isfinite(b) else "" for a, b, c in zip(t["dif"], t["dif_ic_inf"], t["dif_ic_sup"])]
    return t.pivot(index="modelo", columns="metrica", values=["valor_ic", "dif_ic"])


# ---------------------------------------------------------------- por grupo
def tramo_duracion(dur_dias):
    """Tramos de duracion planeada del plan (4.5): <= 3, 3-6, 6-12 y > 12 meses."""
    return pd.cut(pd.Series(np.asarray(dur_dias, dtype="float64")), [-np.inf, 90, 180, 365, np.inf],
                  labels=["<=3 meses", "3-6 meses", "6-12 meses", ">12 meses"]).astype("object").fillna("sin dato").to_numpy()


def evaluar_por_grupo(y_true, score, grupo, fracciones_k=FRACCIONES_K):
    """Metricas por grupo (tramo de duracion, entidad vista/no vista, ...)."""
    y, s, g = np.asarray(y_true), np.asarray(score), np.asarray(grupo)
    filas = []
    for valor in pd.unique(g):
        mask = g == valor
        filas.append({"grupo": valor, **metricas(y[mask], s[mask], fracciones_k)})
    return pd.DataFrame(filas).sort_values("grupo").reset_index(drop=True)


# ---------------------------------------------------------------- evaluacion completa
def puntaje(estimador, X):
    """Probabilidad de la clase positiva o, si el modelo no la da (SVM sin calibrar), la funcion de decision."""
    if hasattr(estimador, "predict_proba"):
        try:
            return estimador.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return estimador.decision_function(X)


def evaluar(estimador, base, anio_test, X=None, etiqueta="y", anio_fin_train=None, m=None, n_boot=200,
            ajustar=True, otros=None):
    """Entrena `estimador` (clon) en el train temporal y lo compara en anio_test con las dos reglas.

    estimador: cualquier objeto con fit/predict_proba que reciba la salida de construir_features.
    X: features ya construidas para toda la base (se calcula si falta). m: suavizado de la regla (si falta, elegir_m
    con folds temporales dentro del train). ajustar=False usa el estimador ya entrenado. otros: dict de estimadores
    ya entrenados para sumar a la comparacion.
    Devuelve {"scores", "tabla", "por_grupo", "m", "estimador", "tabla_m"}.
    """
    X = F.construir_features(base) if X is None else X
    idx_tr, idx_te = split_temporal(base, anio_test, anio_fin_train=anio_fin_train)
    y_tr, y_te = base.loc[idx_tr, etiqueta].to_numpy(), base.loc[idx_te, etiqueta].to_numpy()
    ent_tr, ent_te = base.loc[idx_tr, "nit_entidad"].to_numpy(), base.loc[idx_te, "nit_entidad"].to_numpy()

    tabla_m = None
    if m is None:
        m, tabla_m = elegir_m(ent_tr, y_tr, base.loc[idx_tr, "anio_firma"].to_numpy(), tabla=True)
    est = clone(estimador).fit(X.loc[idx_tr], y_tr) if ajustar else estimador

    scores = {"regla_base": regla_tasa_base(y_tr, len(idx_te)),
              "regla_entidad": regla_tasa_entidad(ent_tr, y_tr, ent_te, m),
              "modelo": puntaje(est, X.loc[idx_te])}
    for nombre, e in (otros or {}).items():
        scores[nombre] = puntaje(e, X.loc[idx_te])
    tabla = comparar(y_te, scores, ent_te, n_boot=n_boot)

    vista = np.where(pd.Series(ent_te).isin(set(ent_tr)), "vista", "no vista")
    tramo = tramo_duracion(base.loc[idx_te, "eval_dur_planeada_dias"])
    por_grupo = pd.concat(
        [evaluar_por_grupo(y_te, s, g).assign(modelo=k, tipo=t)
         for k, s in scores.items() for t, g in (("entidad", vista), ("duracion", tramo))], ignore_index=True)
    sc = pd.DataFrame(scores, index=idx_te).assign(y=y_te, nit_entidad=ent_te, entidad_vista=vista)
    return {"scores": sc, "tabla": tabla, "por_grupo": por_grupo, "m": m, "estimador": est, "tabla_m": tabla_m}
