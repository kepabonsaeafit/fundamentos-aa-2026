"""Monitoreo y reajuste del modelo (G3, requisito 3.9).

Que hace este modulo:
- Mide si las ENTRADAS y el PUNTAJE cambian (PSI), sin esperar etiquetas: se calcula cada mes con TODOS los contratos
  firmados, porque las entradas se conocen al firmar. Calcularlo solo con los contratos maduros sesga los meses
  recientes: los contratos sin duracion planeada no maduran hasta pasado un anio y desaparecen del calculo.
- Mide si el DESEMPENO baja (ROC-AUC, PR-AUC, precision en el top 1 %), cada trimestre y solo con cohortes completas
  (>= 95 % maduras). Se compara con el MISMO trimestre del anio anterior: la tasa base va de 34 % en enero a 16 % en
  octubre y el ROC-AUC mensual oscila de 0,71 a 0,87 solo por la composicion de cada mes, asi que comparar contra un
  promedio anual daria una alarma cada enero.
- Vigila la ETIQUETA: la tasa de positivos frente al mismo trimestre del anio anterior (el salto de y_dias en 2025 es un
  cambio de registro, no de comportamiento).
- Define como se valida y se promueve un modelo nuevo (campeon/retador con IC por entidad).

Todas las reglas estan en las constantes de abajo, con su justificacion en docs/REAJUSTE.md. Los intervalos remuestrean
ENTIDADES, no contratos: los contratos de una entidad estan correlacionados.

Uso (desde final-project/, con la base v1 en data/processed/):  python -m src.monitoreo
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config
from . import features as F

# ---------------------------------------------------------------- reglas
ANIOS_REFERENCIA = (2022, 2023)     # antes de 2021 dias_firma_a_inicio tenia otro regimen (48 % negativos en 2019)
FEATURES_PSI = {                    # entradas con significado de negocio; sin calendario (un mes tiene un solo valor)
    "num": ["log_precio_base", "p_dur_dias", "dias_publicacion_a_firma", "dias_firma_a_inicio", "len_objeto"],
    "cat": ["modalidad_de_contratacion", "tipo_de_contrato", "sector", "origen_de_los_recursos", "unspsc_segmento"],
}
PSI_AVISO, PSI_ALARMA = 0.10, 0.25  # convencion habitual del PSI [SUPUESTO]; se contrasta con la referencia en la simulacion
MIN_VARIABLES_REVISAR = 2             # entradas: se revisa si 2 o mas variables superan PSI_AVISO (o alguna PSI_ALARMA)
MADUREZ_MIN = 0.95                  # fraccion de contratos maduros para evaluar un trimestre con etiquetas
DELTA_TASA_BASE = 0.05              # |cambio| de la tasa de positivos frente al mismo trimestre del anio anterior
DELTA_ROC = 0.03                    # caida minima del ROC-AUC frente al mismo trimestre (ademas de que el IC la excluya)
FRACCION_TOP = 0.01                 # el 1 % de mayor puntaje
CONCENTRACION_TOP = 0.50            # una sola entidad con mas de la mitad del top
PERSISTENCIA = 2                    # periodos seguidos para pasar de aviso a alarma
CORTE_CENSURA = pd.Timestamp("2026-10-26")      # corte de la base v1 (meta.json)
MADURACION_DIAS = 60                # eval_madura: fin planeado + 60 dias (la prorroga se firma antes del vencimiento)
SIN_DURACION_DIAS = 365             # sin duracion planeada: maduro a los 365 dias de la firma

METRICAS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}


# ---------------------------------------------------------------- madurez
def madura_en(base, fecha):
    """eval_madura para cualquier fecha (la regla de meta.json): fin planeado + 60 dias <= fecha; sin duracion, firma + 365."""
    sin_duracion = base["eval_dur_planeada_dias"].fillna(0) <= 0
    fin = (base["eval_fin_planeado"] + pd.Timedelta(days=MADURACION_DIAS)).where(
        ~sin_duracion, base["fecha_de_firma"] + pd.Timedelta(days=SIN_DURACION_DIAS))
    return fin <= pd.Timestamp(fecha)


# ---------------------------------------------------------------- PSI
def _psi(p_ref, p_act, eps=1e-4):
    p_ref, p_act = np.clip(p_ref, eps, None), np.clip(p_act, eps, None)
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))


def psi_numerica(ref, act, bins=10):
    """PSI con los deciles de la referencia como bordes; los faltantes son un grupo aparte."""
    ref, act = pd.Series(ref, dtype="float64"), pd.Series(act, dtype="float64")
    bordes = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    bordes[0], bordes[-1] = -np.inf, np.inf

    def partes(s):
        c = pd.cut(s, bordes, include_lowest=True).value_counts(sort=False).to_numpy().astype(float)
        c = np.append(c, s.isna().sum())
        return c / c.sum()
    return _psi(partes(ref), partes(act))


def psi_categorica(ref, act, top=15):
    """PSI sobre las `top` categorias mas frecuentes de la referencia; el resto (incluidas las nuevas) es '(otras)'."""
    ref = pd.Series(ref).astype("object").fillna("(sin dato)")
    act = pd.Series(act).astype("object").fillna("(sin dato)")
    categorias = list(ref.value_counts().head(top).index)

    def partes(s):
        c = s.where(s.isin(categorias), "(otras)").value_counts()
        v = np.array([c.get(k, 0) for k in categorias + ["(otras)"]], dtype=float)
        return v / v.sum()
    return _psi(partes(ref), partes(act))


def psi_entradas(x_ref, x_act):
    """PSI de cada entrada de FEATURES_PSI entre la referencia y el periodo actual (dos salidas de construir_features)."""
    salida = {c: psi_numerica(x_ref[c], x_act[c]) for c in FEATURES_PSI["num"]}
    salida.update({c: psi_categorica(x_ref[c], x_act[c]) for c in FEATURES_PSI["cat"]})
    return pd.Series(salida)


def nivel_psi(valor):
    return "alarma" if valor >= PSI_ALARMA else "aviso" if valor >= PSI_AVISO else "normal"


# ---------------------------------------------------------------- metricas
def metricas(y, score, entidad=None, fraccion_top=FRACCION_TOP):
    """n, tasa base, puntaje medio, PR-AUC, lift (PR-AUC / tasa base), ROC-AUC, precision en el top y su concentracion."""
    y, s = np.asarray(y), np.asarray(score)
    n = len(y)
    salida = {"n": n, "tasa_base": float(y.mean()), "score_medio": float(s.mean())}
    if len(np.unique(y)) == 2:
        salida["pr_auc"] = float(average_precision_score(y, s))
        salida["lift"] = salida["pr_auc"] / salida["tasa_base"]
        salida["roc_auc"] = float(roc_auc_score(y, s))
    top = np.argsort(-s, kind="stable")[:max(1, int(n * fraccion_top))]
    salida["precision_top"] = float(y[top].mean())
    if entidad is not None:
        salida["concentracion_top"] = float(pd.Series(np.asarray(entidad)[top]).value_counts(normalize=True).iloc[0])
    return salida


# ---------------------------------------------------------------- bootstrap por entidad
def _por_entidad(entidades):
    """Posiciones de cada entidad: (orden, limites) tales que orden[limites[i]:limites[i+1]] son las filas de la entidad i."""
    codigos, unicas = pd.factorize(np.asarray(entidades))
    orden = np.argsort(codigos, kind="stable")
    limites = np.searchsorted(codigos[orden], np.arange(len(unicas) + 1))
    return orden, limites


def _remuestrear(orden, limites, rng):
    elegidas = rng.integers(0, len(limites) - 1, len(limites) - 1)
    return np.concatenate([orden[limites[e]:limites[e + 1]] for e in elegidas])


def _ic(valores):
    return float(np.quantile(valores, 0.025)), float(np.quantile(valores, 0.975))


def diferencia_entre_periodos(a, b, metrica, n_boot=200, semilla=config.SEED):
    """metrica(b) - metrica(a) con IC 95 % remuestreando entidades. a y b: DataFrames (y, score, entidad) de periodos distintos."""
    f = METRICAS[metrica]
    delta = f(b["y"], b["score"]) - f(a["y"], a["score"])
    rng = np.random.default_rng(semilla)
    oa, la = _por_entidad(a["entidad"])
    ob, lb = _por_entidad(b["entidad"])
    ya, sa, yb, sb = (np.asarray(x) for x in (a["y"], a["score"], b["y"], b["score"]))
    deltas = []
    for _ in range(n_boot):
        ia, ib = _remuestrear(oa, la, rng), _remuestrear(ob, lb, rng)
        deltas.append(f(yb[ib], sb[ib]) - f(ya[ia], sa[ia]))
    ic_inf, ic_sup = _ic(deltas)
    return {"delta": float(delta), "ic_inf": ic_inf, "ic_sup": ic_sup}


def diferencia_campeon_retador(y, entidad, score_campeon, score_retador, metrica, n_boot=200, semilla=config.SEED):
    """metrica(retador) - metrica(campeon) sobre los MISMOS contratos, con IC 95 % remuestreando entidades."""
    f = METRICAS[metrica]
    y, sc, sr = np.asarray(y), np.asarray(score_campeon), np.asarray(score_retador)
    delta = f(y, sr) - f(y, sc)
    rng = np.random.default_rng(semilla)
    orden, limites = _por_entidad(entidad)
    deltas = []
    for _ in range(n_boot):
        i = _remuestrear(orden, limites, rng)
        deltas.append(f(y[i], sr[i]) - f(y[i], sc[i]))
    ic_inf, ic_sup = _ic(deltas)
    return {"delta": float(delta), "ic_inf": ic_inf, "ic_sup": ic_sup}


def decision_promocion(pr_auc, roc_auc):
    """Regla de promocion del retador (docs/REAJUSTE.md): el IC 95 % de la mejora en PR-AUC excluye 0 (el principal) y el
    ROC-AUC no empeora mas de 0,01 de forma clara. Recibe los dicts de diferencia_campeon_retador."""
    mejora_clara = pr_auc["ic_inf"] > 0
    roc_empeora_claro = roc_auc["ic_sup"] < -0.01        # todo el IC por debajo de -0,01
    if mejora_clara and not roc_empeora_claro:
        return "promover"
    if pr_auc["ic_sup"] < 0:
        return "descartar: el retador es peor"
    return "no promover: sin evidencia suficiente de mejora"


# ---------------------------------------------------------------- reglas de alarma
def nivel_con_persistencia(avisos, persistencia=PERSISTENCIA):
    """De una serie booleana de avisos: 'alarma' si el aviso se repite `persistencia` periodos seguidos; si no, 'aviso' o 'normal'."""
    avisos = pd.Series(avisos, dtype=bool).reset_index(drop=True)
    racha = avisos.groupby((~avisos).cumsum()).cumsum()
    return ["alarma" if r >= persistencia else "aviso" if a else "normal" for a, r in zip(avisos, racha)]


def psi_historico_entre_anios(base, x, anios=(2021, 2022, 2023)):
    """PSI de cada entrada entre anios consecutivos del entrenamiento, mes a mes: cuanto variaban las entradas cuando todavia
    no habia nada que reajustar. Sirve para ubicar el PSI de la produccion frente a lo que el modelo ya habia visto."""
    mes, anio = base["fecha_de_firma"].dt.month, base["anio_firma"]
    filas = []
    for ref, act in zip(anios[:-1], anios[1:]):
        for m in range(1, 13):
            r, a = ((anio == ref) & (mes == m)).to_numpy(), ((anio == act) & (mes == m)).to_numpy()
            fila = psi_entradas(x[r], x[a]).to_dict()
            fila.update({"mes": m, "par": f"{act} frente a {ref}"})
            filas.append(fila)
    return pd.DataFrame(filas)


def tabla_mensual(base, x, score, anio_ini=2024, anio_fin=2025):
    """Una fila por mes de firma, con TODOS los contratos: tamanio, puntaje, PSI del puntaje frente al mismo mes del anio
    anterior (desde el segundo anio) y PSI de las entradas frente al mismo mes de los anios de referencia."""
    mes = base["fecha_de_firma"].dt.to_period("M")
    ref_mask = base["anio_firma"].isin(ANIOS_REFERENCIA)
    filas = []
    for m in pd.period_range(f"{anio_ini}-01", f"{anio_fin}-12", freq="M"):
        act = (mes == m).to_numpy()
        ref = (ref_mask & (base["fecha_de_firma"].dt.month == m.month)).to_numpy()
        fila = {"mes": str(m), "n": int(act.sum()), "score_medio": float(score[act].mean())}
        previo = (mes == m - 12).to_numpy()
        fila["psi_score_vs_anio_anterior"] = psi_numerica(score[previo], score[act]) if previo.any() and m.year > anio_ini else np.nan
        psi = psi_entradas(x[ref], x[act])
        for c, v in psi.items():
            fila[f"psi_{c}"] = v
        fila["psi_max"], fila["variable_psi_max"] = float(psi.max()), psi.idxmax()
        fila["n_variables_psi_alto"] = int((psi >= PSI_AVISO).sum())
        filas.append(fila)
    t = pd.DataFrame(filas)
    # Las entradas nunca reentrenan por si solas: el maximo de 10 variables avisa casi siempre (y entre anios consecutivos
    # del entrenamiento el PSI ya era enorme, ver psi_historico_entre_anios), asi que se pide mas de una variable.
    t["estado_entradas"] = np.where((t["n_variables_psi_alto"] >= MIN_VARIABLES_REVISAR) | (t["psi_max"] >= PSI_ALARMA),
                                    "revisar", "normal")
    t["estado_score"] = [nivel_psi(v) if pd.notna(v) else "sin referencia" for v in t["psi_score_vs_anio_anterior"]]
    return t


def tabla_trimestral(base, score, corte=CORTE_CENSURA, n_boot=200, anio_ini=2024, anio_fin=2025):
    """Una fila por trimestre de firma. Todo lo que usa etiquetas (tasas, desempeno) se calcula solo si el trimestre esta
    completo (>= MADUREZ_MIN maduros), con los contratos maduros, y se compara con el MISMO trimestre del anio anterior
    solo si ese tambien esta completo: con una cohorte incompleta los pocos maduros estan sesgados por la censura. El
    desempeno se compara desde 2025 (el 2023 es de entrenamiento y su puntaje es optimista); la tasa base, desde 2024."""
    d = pd.DataFrame({"y": base["y"].to_numpy(), "y_dias": base["y_dias"].to_numpy(), "score": score,
                      "entidad": base["nit_entidad"].to_numpy(), "madura": madura_en(base, corte).to_numpy(),
                      "trim": base["fecha_de_firma"].dt.to_period("Q").to_numpy()}, index=base.index)
    filas, maduros, completos = [], {}, {}
    for q in pd.period_range(f"{anio_ini - 1}Q1", f"{anio_fin}Q4", freq="Q"):
        g = d[d["trim"] == q]
        maduros[q], completos[q] = g[g["madura"]], bool(g["madura"].mean() >= MADUREZ_MIN)
        if q.year < anio_ini:
            continue
        gm = maduros[q]
        fila = {"trimestre": str(q), "n": len(g), "madurez": float(g["madura"].mean()), "completo": completos[q]}
        if completos[q]:
            previo = maduros[q - 4]
            fila["tasa_y"], fila["tasa_y_dias"] = float(gm["y"].mean()), float(gm["y_dias"].mean())
            if completos[q - 4]:
                fila["delta_tasa_y"] = fila["tasa_y"] - float(previo["y"].mean())
                fila["delta_tasa_y_dias"] = fila["tasa_y_dias"] - float(previo["y_dias"].mean())
            fila.update({k: v for k, v in metricas(gm["y"], gm["score"], gm["entidad"]).items() if k not in ("n", "tasa_base")})
            if q.year > anio_ini and completos[q - 4]:
                for met in ("roc_auc", "pr_auc"):
                    r = diferencia_entre_periodos(previo, gm, met, n_boot=n_boot)
                    fila[f"delta_{met}"], fila[f"ic_inf_{met}"], fila[f"ic_sup_{met}"] = r["delta"], r["ic_inf"], r["ic_sup"]
        filas.append(fila)
    t = pd.DataFrame(filas)
    for c in ("tasa_y", "tasa_y_dias", "delta_tasa_y", "delta_tasa_y_dias", "precision_top", "concentracion_top",
              "delta_roc_auc", "ic_sup_roc_auc"):
        if c not in t:
            t[c] = np.nan
    t["aviso_tasa_y"] = t["delta_tasa_y"].abs() >= DELTA_TASA_BASE
    t["aviso_tasa_y_dias"] = t["delta_tasa_y_dias"].abs() >= DELTA_TASA_BASE
    t["aviso_concentracion"] = t["concentracion_top"] > CONCENTRACION_TOP
    # la lista de mayor riesgo no puede ser peor que un contrato al azar (precision del top < tasa base)
    t["aviso_top_peor_que_base"] = t["precision_top"] < t["tasa_y"]
    t["estado_top"] = nivel_con_persistencia(t["aviso_top_peor_que_base"])
    t["aviso_desempeno"] = (t["delta_roc_auc"] <= -DELTA_ROC) & (t["ic_sup_roc_auc"] < 0)
    t["estado_desempeno"] = nivel_con_persistencia(t["aviso_desempeno"])
    return t


def entrenar_modelo(base, hasta_anio, construir_pipeline):
    """Pipeline nuevo ajustado con los contratos maduros firmados de 2019 a `hasta_anio` (etiqueta y)."""
    train = base[(base["anio_firma"] >= config.ANIO_INICIO) & (base["anio_firma"] <= hasta_anio) & base["eval_madura"]]
    return construir_pipeline().fit(F.construir_features(train), train["y"])


def simular(base, construir_pipeline, n_boot=200, log=print, x=None):
    """Recorre 2024-2025 con el modelo congelado (entrenado en 2019-2023) y devuelve (mensual, trimestral, campeon_retador).

    [SUPUESTO] El modelo se entrena con las etiquetas finales de 2019-2023; en produccion real solo habria visto las
    disponibles el dia de la puesta en marcha. El retador se entrena con 2019-2024 y se evalua en los trimestres
    completos de 2025, que ninguno de los dos vio.
    """
    t0 = time.perf_counter()
    x = F.construir_features(base) if x is None else x
    campeon = entrenar_modelo(base, config.ANIO_TEST - 1, construir_pipeline)
    score = np.full(len(base), np.nan)          # solo 2024 en adelante: el puntaje en muestra del entrenamiento es optimista
    posteriores = (base["anio_firma"] >= config.ANIO_TEST).to_numpy()
    score[posteriores] = campeon.predict_proba(x[posteriores])[:, 1]
    log(f"modelo congelado listo ({time.perf_counter() - t0:.0f} s)")

    mensual = tabla_mensual(base, x, score)
    trimestral = tabla_trimestral(base, score, n_boot=n_boot)
    log(f"tablas listas ({time.perf_counter() - t0:.0f} s)")

    retador = entrenar_modelo(base, config.ANIO_DERIVA - 1, construir_pipeline)
    completos = set(trimestral.loc[trimestral["completo"] & trimestral["trimestre"].str.startswith(str(config.ANIO_DERIVA)),
                                   "trimestre"])
    en_2025 = (base["fecha_de_firma"].dt.to_period("Q").astype(str).isin(completos) & madura_en(base, CORTE_CENSURA)).to_numpy()
    y, ent = base.loc[en_2025, "y"], base.loc[en_2025, "nit_entidad"]
    s_camp = score[en_2025]
    s_ret = retador.predict_proba(x[en_2025])[:, 1]
    pr = diferencia_campeon_retador(y, ent, s_camp, s_ret, "pr_auc", n_boot=n_boot)
    roc = diferencia_campeon_retador(y, ent, s_camp, s_ret, "roc_auc", n_boot=n_boot)
    cr = {"n_evaluacion": int(en_2025.sum()), "trimestres": sorted(completos),
          "pr_auc_campeon": float(average_precision_score(y, s_camp)), "pr_auc_retador": float(average_precision_score(y, s_ret)),
          "roc_auc_campeon": float(roc_auc_score(y, s_camp)), "roc_auc_retador": float(roc_auc_score(y, s_ret)),
          "delta_pr_auc": pr, "delta_roc_auc": roc, "decision": decision_promocion(pr, roc)}
    log(f"campeon/retador listo ({time.perf_counter() - t0:.0f} s)")
    return mensual, trimestral, cr


def resumen_markdown(mensual, trimestral, cr, historico=None):
    """Resumen de la simulacion para resultados/monitoreo_resumen.md."""
    def lista(serie):
        return ", ".join(serie) if len(serie) else "ninguno"
    ent = mensual["estado_entradas"].value_counts()
    sco = mensual["estado_score"].value_counts()
    completos = trimestral[trimestral["completo"]]
    lineas = [
        "# Simulacion de monitoreo 2024-2025",
        "",
        "Modelo congelado: entrenado con 2019-2023 (contratos maduros, etiqueta `y`), sin reentrenar. Base v1, corte 2026-10-26.",
        "Se generan con `python -m src.monitoreo`. Reglas y justificacion en `docs/REAJUSTE.md`.",
        "",
        "## Entradas y puntaje (mensual, todos los contratos)",
        f"- Entradas frente al mismo mes de {ANIOS_REFERENCIA[0]}-{ANIOS_REFERENCIA[1]}: {ent.get('normal', 0)} meses normales y "
        f"{ent.get('revisar', 0)} para revisar (2 o mas de las 10 variables con PSI >= {PSI_AVISO}, o alguna >= {PSI_ALARMA}).",
        f"- Meses para revisar: {lista(mensual.loc[mensual['estado_entradas'] == 'revisar', 'mes'])}.",
        f"- Puntaje frente al mismo mes del anio anterior (2025): {sco.get('normal', 0)} meses normales, "
        f"{sco.get('aviso', 0)} en aviso, {sco.get('alarma', 0)} en alarma. PSI maximo: {mensual['psi_score_vs_anio_anterior'].max():.3f}.",
    ]
    if historico is not None:
        cols = FEATURES_PSI["num"] + FEATURES_PSI["cat"]
        hist, prod = historico[cols].stack(), mensual[["psi_" + c for c in cols]].stack()
        lineas.append(
            f"- Contexto: entre anios consecutivos del entrenamiento (2021-2023) el PSI de las entradas ya tenia mediana "
            f"{hist.median():.3f} y el {(hist >= PSI_ALARMA).mean():.0%} de los valores superaba {PSI_ALARMA}; en 2024-2025 la mediana es "
            f"{prod.median():.3f} y el {(prod >= PSI_ALARMA).mean():.0%} supera {PSI_ALARMA}. El PSI avisa de un cambio, pero no dice "
            "si el modelo se degrado: por eso no dispara el reentrenamiento por si solo.")
    lineas += [
        "",
        "## Desempeno y etiqueta (trimestral, solo cohortes completas)",
        f"- Trimestres completos (>= {MADUREZ_MIN:.0%} maduros): {lista(completos['trimestre'])}.",
        f"- Trimestres incompletos, sin evaluar con etiquetas: {lista(trimestral.loc[~trimestral['completo'], 'trimestre'])}.",
        f"- Aviso por cambio de la tasa de `y` (>= {DELTA_TASA_BASE:.0%} frente al mismo trimestre): {lista(trimestral.loc[trimestral['aviso_tasa_y'], 'trimestre'])}.",
        f"- Aviso por cambio de la tasa de `y_dias` (>= {DELTA_TASA_BASE:.0%}): {lista(trimestral.loc[trimestral['aviso_tasa_y_dias'], 'trimestre'])}.",
        f"- Top {FRACCION_TOP:.0%} peor que un contrato al azar (precision < tasa base): {lista(trimestral.loc[trimestral['aviso_top_peor_que_base'], 'trimestre'])}.",
        f"- Top {FRACCION_TOP:.0%} dominado por una sola entidad (> {CONCENTRACION_TOP:.0%}), informativo: {lista(trimestral.loc[trimestral['aviso_concentracion'], 'trimestre'])}.",
    ]
    lineas.append(f"- Alarma de desempeno (ROC-AUC cae >= {DELTA_ROC} frente al mismo trimestre, con IC 95 % que excluye 0, "
                  f"{PERSISTENCIA} trimestres seguidos): {lista(trimestral.loc[trimestral['estado_desempeno'] == 'alarma', 'trimestre'])}.")
    lineas += [
        "",
        "## Campeon frente a retador (2019-2023 frente a 2019-2024, evaluados en 2025)",
        f"- Contratos de evaluacion: {cr['n_evaluacion']:,}".replace(",", ".") + f" ({', '.join(cr['trimestres'])}).",
        f"- PR-AUC: campeon {cr['pr_auc_campeon']:.3f}, retador {cr['pr_auc_retador']:.3f}; diferencia {cr['delta_pr_auc']['delta']:+.3f} "
        f"[{cr['delta_pr_auc']['ic_inf']:+.3f}, {cr['delta_pr_auc']['ic_sup']:+.3f}].",
        f"- ROC-AUC: campeon {cr['roc_auc_campeon']:.3f}, retador {cr['roc_auc_retador']:.3f}; diferencia {cr['delta_roc_auc']['delta']:+.3f} "
        f"[{cr['delta_roc_auc']['ic_inf']:+.3f}, {cr['delta_roc_auc']['ic_sup']:+.3f}].",
        f"- Decision segun la regla de promocion: **{cr['decision']}**.",
    ]
    return "\n".join(lineas) + "\n"


def main(n_boot=200):
    from . import modelo_dummy
    base = pd.read_parquet(config.DATA_PROCESSED / "base_contratos.parquet")
    # comprobacion previa: la regla de madurez reproduce la columna de la base
    assert (madura_en(base, CORTE_CENSURA) == base["eval_madura"]).all(), "madura_en no reproduce eval_madura"
    x = F.construir_features(base)
    historico = psi_historico_entre_anios(base, x)
    mensual, trimestral, cr = simular(base, modelo_dummy.construir_pipeline, n_boot=n_boot, x=x)
    destino = config.RAIZ / "resultados"
    destino.mkdir(exist_ok=True)
    mensual.to_csv(destino / "monitoreo_mensual.csv", index=False)
    trimestral.to_csv(destino / "monitoreo_trimestral.csv", index=False)
    historico.to_csv(destino / "monitoreo_psi_historico.csv", index=False)
    (destino / "monitoreo_campeon_retador.json").write_text(json.dumps(cr, indent=2, ensure_ascii=False), encoding="utf-8")
    (destino / "monitoreo_resumen.md").write_text(resumen_markdown(mensual, trimestral, cr, historico), encoding="utf-8")
    print((destino / "monitoreo_resumen.md").read_text(encoding="utf-8"))
    return mensual, trimestral, cr, historico


if __name__ == "__main__":
    main()
