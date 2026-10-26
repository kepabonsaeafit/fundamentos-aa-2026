"""Reporte de calidad de la base v1 (PLAN.md 4.7). Es lo que aprueba el usuario; no oculta fallas.

Cada chequeo tiene estado 'ok', 'falla' o 'no aplica', su evidencia y si bloquea (un chequeo no bloqueante que falla se muestra
igual, con la razon). Modo A: la 4.7 completa. Modo B: tamano por anio y digito 'no aplica'; el chequeo 3 (tasa de y_dias por anio a
+-1,5 pp de la poblacion) en version PONDERADA por mes (bloqueante) y en version cruda (no bloqueante: falla en 2019 por el
diseno de la muestra, muestra_exploracion.md 3.7); distribucion de contratos y de positivos por anio frente a la poblacion.

evaluar(base, X, ctx) -> dict;  escribir(resultado, carpeta, etapas) -> reporte_calidad.md y reporte_calidad.json.
"""
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import base as B
from . import config
from . import features as F
from . import pesos as P

TOL_TAMANO = 0.02            # tamano por anio = 10 % de la poblacion, +-2 % relativo
TOL_Y_DIAS_PP = 1.5          # tasa de y_dias por anio frente a la poblacion
BANDA_Y = (19.0, 26.0)       # tasa de la etiqueta por anio 2019-2024 (esperada ~24 %)
AUC_MAX = 0.85
DUR_MIN, DUR_MAX = 1, 3650   # dias
# Tasas de referencia de la v0 (data/processed/base_contratos.parquet, 167.997 filas); bloquean si la v1 supera el doble
TASA_NEG_V0 = 26 / 167_997            # 0,0155 %: 26 valores negativos, todos en p_precio_base
TASA_DUR_FUERA_V0 = 8_760 / 167_997   # 5,2144 %: 8.694 con duracion 0 (< DUR_MIN) y 66 > DUR_MAX
FACTOR_ALERTA_V0 = 2.0
MIN_CON_PROCESO = 0.95
CONTEO_DIG4 = {2019: 14407, 2020: 36107, 2021: 56469, 2022: 71095, 2023: 83764, 2024: 88548, 2025: 102125}   # PLAN.md 4.1
ANIOS = list(range(config.ANIO_INICIO, config.ANIO_DERIVA + 1))

# Referencias de .agents/tasks/f1/muestra_exploracion.md (B: misma foto del 14-oct, asi que la base debe reproducirlas)
REF_Y_DIAS_MUESTRA_PCT = {2019: 9.34, 2020: 12.79, 2021: 10.03, 2022: 10.41, 2023: 12.09, 2024: 12.26, 2025: 19.59}     # 3.7
REF_Y_DIAS_POND_PCT = {2019: 10.68, 2020: 12.97, 2021: 9.78, 2022: 9.40, 2023: 12.75, 2024: 12.14, 2025: 20.05}         # 3.7
REF_DIF_CRUDA_PP = {2019: -1.76, 2020: -0.32, 2021: 0.34, 2022: 0.87, 2023: -0.75, 2024: 0.13, 2025: -0.44}             # 3.7
REF_COMPOSICION = {"Contratación directa": (65.04, 76.77), "Contratación régimen especial": (13.88, 13.28),
                   "Mínima cuantía": (12.49, 5.43)}                                                                      # 3.8 (cruda, ponderada)
REF_DIRECTA_ANIO = {2019: (24.2, 61.3), 2022: (56.5, 80.1)}                                                              # 3.8


def _es(s):
    """Formato español de los números calculados (escritos como 12,345.6): miles con punto y decimales con coma.
    Las referencias a secciones se escriben '§3.7' (el § las protege) y los números escritos a mano ya vienen en español."""
    s = re.sub(r"(?<![§\d.])(\d+)\.(\d+)", "\\1\x00\\2", s)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", ".", s)
    return s.replace("\x00", ",")


def chequeo(id_, nombre, estado, evidencia, bloquea=True, tabla=None, nota=None):
    assert estado in ("ok", "falla", "no aplica")
    return {"id": id_, "chequeo": nombre, "estado": estado, "bloquea": bool(bloquea), "evidencia": _es(evidencia),
            "tabla": tabla, "nota": _es(nota) if nota else None}


def _recs(df):
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _pct(x, d=2):
    return round(100 * float(x), d)


def _tasa_ponderada(y, w, por):
    g = pd.DataFrame({"y": np.asarray(y, dtype="float64"), "w": np.asarray(w, dtype="float64"), "k": np.asarray(por)})
    g["yw"] = g["y"] * g["w"]
    s = g.groupby("k")[["yw", "w"]].sum()
    return s["yw"] / s["w"]


# ---------------------------------------------------------------- AUC univariado
def auc_univariado(X, y, semilla=config.SEED, k=5, m=20):
    """ROC-AUC univariado de cada feature contra y. Numericas: el valor (sin NaN); categoricas: codificacion por media de y fuera de
    muestra (5 pliegues, suavizado m=20). Se devuelve max(auc, 1 - auc)."""
    y = np.asarray(y, dtype="float64")
    pliegue = np.random.default_rng(semilla).integers(0, k, len(y))
    p0 = y.mean()
    out = {}
    for c in X.columns:
        s = X[c]
        if s.dtype == object:
            codigos, _ = pd.factorize(s, use_na_sentinel=False)
            n_cod = int(codigos.max()) + 1
            enc = np.empty(len(y))
            for f in range(k):
                tr = pliegue != f
                suma = np.bincount(codigos[tr], weights=y[tr], minlength=n_cod)
                cuenta = np.bincount(codigos[tr], minlength=n_cod)
                enc[~tr] = (suma[codigos[~tr]] + m * p0) / (cuenta[codigos[~tr]] + m)
            auc = roc_auc_score(y, enc)
        else:
            v = s.to_numpy("float64")
            ok = ~np.isnan(v)
            auc = roc_auc_score(y[ok], v[ok]) if ok.sum() > 1 and len(np.unique(v[ok])) > 1 and len(np.unique(y[ok])) == 2 else 0.5
        out[c] = max(auc, 1 - auc)
    return pd.Series(out).sort_values(ascending=False)


def _error_texto(e):
    if not isinstance(e, dict):
        return str(e)
    if e.get("tipo") == "exacto":
        return f"exacto ({e['error_relativo_pct']})"
    return (f"estimado, mediana {e['error_relativo_mediano_pct']} %, máximo {e['error_relativo_maximo_pct']} % (mes {e['mes_de_mayor_error']}), "
            f"regla {e['regla']}")


# ---------------------------------------------------------------- evaluacion
def evaluar(base, X, ctx):
    modo = ctx["modo"]
    A, Bm = modo == "A", modo == "B"
    pob = P.poblacion_anual()
    pob_n = pob["n"].reindex(ANIOS)
    pob_tasa = 100 * pob["con_dias_adicionados"].reindex(ANIOS) / pob_n
    anio = base["anio_firma"]
    w = base["peso_mes"].to_numpy("float64")
    y_dias, y = base["y_dias"].to_numpy(), base["y"].to_numpy()
    por_anio = base.groupby("anio_firma").size().reindex(ANIOS)
    corte = pd.Timestamp(ctx["corte"])
    digito = str(ctx.get("digito") or "4")
    checks, info = [], {}

    # 1. id_contrato unico
    dup = int(base["id_contrato"].duplicated().sum())
    checks.append(chequeo(1, "id_contrato único", "ok" if dup == 0 else "falla",
                          f"{len(base):,} filas, {base['id_contrato'].nunique():,} ids únicos, {dup} repetidos; "
                          f"{ctx['contratos']['duplicados_exactos']} filas exactamente duplicadas eliminadas al leer las unidades"))
    # 2. el join no duplica
    j = ctx["join"]
    ok = j["filas_base"] + j["fuera_de_2019_2025_o_sin_fecha"] == j["contratos_entrada"] and dup == 0
    checks.append(chequeo(2, "El join Contratos + Procesos no duplica contratos", "ok" if ok else "falla",
                          f"{j['contratos_entrada']:,} contratos de entrada = {j['filas_base']:,} en la base + "
                          f"{j['fuera_de_2019_2025_o_sin_fecha']:,} fuera de 2019-2025 o sin fecha de firma; "
                          f"uniones.unir_contrato_proceso falla si la unión cambia el número de filas. Procesos: "
                          f"{ctx['procesos']['filas']:,} filas, {ctx['procesos']['portafolios']:,} portafolios, "
                          f"{ctx['procesos']['portafolios_con_varias_filas']:,} con varias filas (elegir_proceso deja una)"))
    # 3. sin PII
    cols_pii = sorted((set(base.columns) | set(ctx["columnas_leidas"]["contratos"]) | set(ctx["columnas_leidas"]["procesos"])
                       | set(ctx["columnas_leidas"]["modificaciones"])) & config.PII)
    checks.append(chequeo(3, "Ninguna columna de datos personales (config.PII)", "ok" if not cols_pii else "falla",
                          f"{len(config.PII)} columnas PII; en la base y en las columnas leídas de las unidades: {cols_pii or 'ninguna'}"))
    # 4. FEATURES y fuga
    inter = sorted(set(F.FEATURES) & F.COLUMNAS_FUGA)
    inter_ent = sorted(set(F.ENTRADAS) & F.COLUMNAS_FUGA)
    inter_base = sorted((set(base.columns) & F.COLUMNAS_FUGA) - set(B.ETIQUETAS))
    checks.append(chequeo(4, "FEATURES ∩ COLUMNAS_FUGA = ∅", "ok" if not (inter or inter_ent or inter_base) else "falla",
                          f"{len(F.FEATURES)} features, {len(F.COLUMNAS_FUGA)} columnas de fuga; intersección con FEATURES {inter or '∅'}, "
                          f"con las entradas {inter_ent or '∅'}, con las columnas de la base (salvo y, y_dias, y_modif) {inter_base or '∅'}"))

    # 5-7. tamano por anio, completitud anio x digito, digito (solo A); cuota por mes (solo B)
    dif = (por_anio / (pob_n / 10) - 1)
    t = pd.DataFrame({"anio": ANIOS, "filas": por_anio.to_numpy(), "10 % de la población": (pob_n / 10).round(0).to_numpy(),
                      "desvío relativo %": (100 * dif).round(2).to_numpy()})
    if A:
        ok = bool((dif.abs() <= TOL_TAMANO).all())
        checks.append(chequeo(5, "Tamaño por año = 10 % de la población (±2 %)", "ok" if ok else "falla",
                              f"desvío máximo {100 * dif.abs().max():.2f} % (tolerancia {100 * TOL_TAMANO:.0f} %); población: "
                              f"poblacion_contratos_por_anio.csv (14-oct)", tabla=_recs(t)))
        if digito == "4":
            esp = pd.Series(CONTEO_DIG4)
            tol = np.maximum(10, np.ceil(0.001 * esp))
            d = por_anio - esp
            ok = bool((d.abs() <= tol).all())
            tt = pd.DataFrame({"anio": ANIOS, "filas": por_anio.to_numpy(), "conteo exacto dígito 4 (24-oct)": esp.to_numpy(),
                               "diferencia": d.to_numpy(), "tolerancia": tol.to_numpy()})
            checks.append(chequeo(6, "Completitud contra el conteo exacto por año × dígito 4", "ok" if ok else "falla",
                                  f"diferencia máxima {int(d.abs().max())} filas (tolerancia max(10, 0,1 %)); detecta lotes que faltan",
                                  tabla=_recs(tt)))
        else:
            checks.append(chequeo(6, "Completitud contra el conteo exacto por año × dígito 4", "no aplica",
                                  f"la tabla de PLAN.md 4.1 es del dígito 4 y esta corrida es del dígito {digito}"))
        if len(digito) == 1 and digito.isdigit():
            malos = int((base["id_contrato"].str[-1] != digito).sum())
            checks.append(chequeo(7, f"Todos los id_contrato terminan en el dígito {digito}", "ok" if malos == 0 else "falla",
                                  f"{malos} filas con otro dígito"))
        else:
            checks.append(chequeo(7, "Todos los id_contrato terminan en el dígito de la muestra", "no aplica",
                                  f"completitud.json no declara un dígito único (digito = {digito!r})"))
        checks.append(chequeo(8, "Cuota de 2.000 contratos por mes (asignación igual)", "no aplica", "es el diseño de B, no de A"))
    else:
        for i, nom in [(5, "Tamaño por año = 10 % de la población (±2 %)"), (6, "Completitud contra el conteo exacto por año × dígito 4"),
                       (7, "Todos los id_contrato terminan en el dígito de la muestra")]:
            checks.append(chequeo(i, nom, "no aplica", "la muestra de B es estratificada por mes (2,000 por mes), no un 10 % por dígito"))
        n_mes = base.groupby(base["fecha_de_firma"].dt.strftime("%Y-%m")).size()
        ok = len(n_mes) == 84 and int(n_mes.min()) >= 1990 and int(n_mes.max()) <= 2000
        checks.append(chequeo(8, "Cuota de 2.000 contratos por mes (asignación igual)", "ok" if ok else "falla",
                              f"{len(n_mes)} meses (se esperan 84); filas por mes: mínimo {int(n_mes.min())}, máximo {int(n_mes.max())} "
                              "(2,000 menos los repetidos que entregó el servidor)"))
    info["tamano_por_anio"] = _recs(t)

    # 9. chequeo 3 de la 4.7: tasa de y_dias por anio frente a la poblacion
    cruda = base.groupby("anio_firma")["y_dias"].mean().reindex(ANIOS) * 100
    ponde = _tasa_ponderada(y_dias, w, anio.to_numpy()).reindex(ANIOS) * 100
    tabla_y = pd.DataFrame({"anio": ANIOS, "n": por_anio.to_numpy(), "y_dias muestra %": cruda.round(2).to_numpy(),
                            "y_dias ponderada %": ponde.round(2).to_numpy(), "y_dias población %": pob_tasa.round(2).to_numpy(),
                            "dif cruda pp": (cruda - pob_tasa).round(2).to_numpy(), "dif ponderada pp": (ponde - pob_tasa).round(2).to_numpy()})
    if Bm:
        ok_p = bool(((ponde - pob_tasa).abs() <= TOL_Y_DIAS_PP).all())
        checks.append(chequeo(9, "Tasa de y_dias por año a ±1,5 pp de la población, PONDERADA por mes", "ok" if ok_p else "falla",
                              f"diferencia ponderada máxima {(ponde - pob_tasa).abs().max():.2f} pp (tolerancia {TOL_Y_DIAS_PP} pp); "
                              "población: poblacion_contratos_por_anio.csv (14-oct, misma foto que la muestra)", tabla=_recs(tabla_y)))
        ok_c = bool(((cruda - pob_tasa).abs() <= TOL_Y_DIAS_PP).all())
        mal = [int(a) for a in ANIOS if abs(cruda[a] - pob_tasa[a]) > TOL_Y_DIAS_PP]
        checks.append(chequeo("9b", "Tasa de y_dias por año a ±1,5 pp de la población, CRUDA (sin pesos)", "ok" if ok_c else "falla",
                              f"diferencia cruda máxima {(cruda - pob_tasa).abs().max():.2f} pp; años fuera de ±1,5 pp: {mal or 'ninguno'}",
                              bloquea=False,
                              nota="Falla esperada por el diseño de B: la muestra da 2,000 contratos a cada mes y 2019 tiene enero y febrero "
                                   "con el 67 % de sus contratos pero solo el 17 % de la muestra (muestra_exploracion.md, §3.6 y §3.7: 2019 queda en "
                                   "-1,76 pp). Con pesos por mes baja a -0,42 pp. No bloquea porque el chequeo vigente en B es el ponderado."))
    else:
        ok = bool(((cruda - pob_tasa).abs() <= TOL_Y_DIAS_PP).all())
        checks.append(chequeo(9, "Tasa de y_dias por año a ±1,5 pp de la población (sin pesos)", "ok" if ok else "falla",
                              f"diferencia máxima {(cruda - pob_tasa).abs().max():.2f} pp (tolerancia {TOL_Y_DIAS_PP} pp); "
                              "población: poblacion_contratos_por_anio.csv (14-oct)", tabla=_recs(tabla_y)))
    info["tasa_y_dias_por_anio"] = _recs(tabla_y)

    # 10. rangos de fechas, montos y duracion
    f_firma = base["fecha_de_firma"]
    fuera_firma = int(((f_firma < "2019-01-01") | (f_firma >= "2026-01-01")).sum())
    futuras = int((f_firma > corte).sum())
    fechas = {c: (base[c].min(), base[c].max(), int(base[c].isna().sum())) for c in ["fecha_de_firma", "fecha_de_inicio_del_contrato",
                                                                                  "p_fecha_de_publicacion_del", "eval_fin_planeado"]}
    directas = ["fecha_de_firma", "fecha_de_inicio_del_contrato", "p_fecha_de_publicacion_del"]     # eval_fin_planeado se deriva de la duración (chequeo 12)
    fuera_rango = {c: int(((base[c] < F.FECHA_MIN) | (base[c] >= F.FECHA_MAX)).sum()) for c in directas}
    ok = fuera_firma == 0 and futuras == 0 and not any(fuera_rango.values())
    checks.append(chequeo(10, "Rangos de fechas (años 0025 y 8201 → NaT)", "ok" if ok else "falla",
                          f"firma entre {f_firma.min():%Y-%m-%d} y {f_firma.max():%Y-%m-%d}; {futuras} firmas posteriores al corte; fechas fuera de "
                          f"[2000, 2100) en la base: {sum(fuera_rango.values())}. NaT por fecha (las mal digitadas pasan a NaT): "
                          + ", ".join(f"{c} {v[2]:,}" for c, v in fechas.items() if c in directas)
                          + f"; eval_fin_planeado {fechas['eval_fin_planeado'][2]:,}, de los cuales "
                          f"{int((base['eval_fin_planeado'] >= F.FECHA_MAX).sum())} caen después de 2100 por duraciones absurdas (chequeo 12)"))
    neg = {c: int((base[c] < 0).sum()) for c in B._NUMEROS}
    neg = {c: v for c, v in neg.items() if v}
    neg_total = sum(neg.values())
    tasa_neg = neg_total / len(base)
    checks.append(chequeo(11, "Montos y conteos ≥ 0", "ok" if not neg else "falla",
                          f"valores negativos por columna numérica: {neg or 'ninguno'} (columnas: valor_de_pago_adelantado, p_precio_base y los conteos de Procesos); "
                          f"tasa {100 * tasa_neg:.4f} % frente a {100 * TASA_NEG_V0:.4f} % en v0; bloquea si supera el doble",
                          bloquea=tasa_neg > FACTOR_ALERTA_V0 * TASA_NEG_V0,
                          nota=("Dato de origen: son errores de digitación del portal, no de la construcción. La base los conserva tal cual y "
                                "construir_features recorta los montos en 0 (log1p con clip inferior). Decisión del usuario si se corrigen en la base.")
                          if neg else None))
    dur = base["eval_dur_planeada_dias"].astype("float64")
    menor, mayor = int((dur < DUR_MIN).sum()), int((dur > DUR_MAX).sum())
    fuera_dur = menor + mayor
    tasa_dur = fuera_dur / len(base)
    checks.append(chequeo(12, "Duración planeada entre 1 y 3.650 días", "ok" if fuera_dur == 0 else "falla",
                          f"{fuera_dur:,} contratos fuera de [{DUR_MIN}, {DUR_MAX}] días ({100 * fuera_dur / len(base):.2f} %): {menor:,} con menos de {DUR_MIN} día "
                          f"y {mayor:,} con más de {DUR_MAX}; {int(dur.isna().sum()):,} sin duración ({100 * dur.isna().mean():.2f} %, maduran por la regla de respaldo: firma + 365 días); "
                          f"mediana {dur.median():.0f}, p99 {dur.quantile(.99):.0f}, máximo {dur.max():.0f}; "
                          f"tasa {100 * tasa_dur:.2f} % frente a {100 * TASA_DUR_FUERA_V0:.2f} % en v0; bloquea si supera el doble",
                          bloquea=tasa_dur > FACTOR_ALERTA_V0 * TASA_DUR_FUERA_V0,
                          nota=("Dato de origen: el rango 1-3,650 días es la regla de limpieza de la exploración (se usa para excluir duraciones absurdas, "
                                "unidad mal digitada). La base no borra esos contratos: eval_fin_planeado recorta la duración a 36,500 días y los de más "
                                "de 3,650 días quedan sin madurar por años (no entran a train ni test); los de duración 0 maduran por la regla de respaldo (firma + 365 días). "
                                "Decisión del usuario si se excluyen o se marcan.") if fuera_dur else None))

    # 13. proceso, faltantes y dtypes
    con_proc = float(base["eval_en_procesos"].mean())
    checks.append(chequeo(13, "≥ 95 % de los contratos con proceso", "ok" if con_proc >= MIN_CON_PROCESO else "falla",
                          f"{100 * con_proc:.2f} % con proceso ({int((~base['eval_en_procesos']).sum()):,} sin proceso)"))
    falt = X.isna().mean().mul(100).round(2)
    esquema_ok, esquema_msg = True, "esquema 2a validado (columnas, orden y dtypes)"
    try:
        B.validar_esquema(base)
    except ValueError as e:
        esquema_ok, esquema_msg = False, str(e)
    tipos_ok = dict(X.dtypes.astype(str)) == {c: F._TIPO[c] for c in F.FEATURES} and list(X.columns) == F.FEATURES
    todo_nulo = sorted(falt[falt >= 100].index)
    checks.append(chequeo(14, "Faltantes por feature y dtypes iguales al esquema", "ok" if esquema_ok and tipos_ok and not todo_nulo else "falla",
                          f"{esquema_msg}; construir_features devuelve exactamente FEATURES con sus dtypes: {tipos_ok}; "
                          f"features 100 % faltantes: {todo_nulo or 'ninguna'}; mayor faltante: {falt.idxmax()} {falt.max():.1f} %",
                          tabla=_recs(falt.sort_values(ascending=False).head(12).rename("faltante %").reset_index().rename(columns={"index": "feature"}))))
    info["faltantes_por_feature_pct"] = {k: float(v) for k, v in falt.sort_values(ascending=False).items()}

    # 15. tasa de la etiqueta por anio
    y_pond = _tasa_ponderada(y, w, anio.to_numpy()).reindex(ANIOS) * 100
    y_cruda = base.groupby("anio_firma")["y"].mean().reindex(ANIOS) * 100
    ymod = base.groupby("anio_firma")["y_modif"].apply(lambda s: float(s.astype("float64").mean())).reindex(ANIOS) * 100
    ref = y_pond if Bm else y_cruda
    tabla_e = pd.DataFrame({"anio": ANIOS, "y muestra %": y_cruda.round(2).to_numpy(), "y ponderada %": y_pond.round(2).to_numpy(),
                            "y_dias %": cruda.round(2).to_numpy(), "y_modif %": ymod.round(2).to_numpy()})
    en_banda = ref.loc[2019:2024].between(*BANDA_Y)
    checks.append(chequeo(15, f"Tasa de la etiqueta por año en {BANDA_Y[0]:.0f}-{BANDA_Y[1]:.0f} % (2019-2024; esperada ~24 %)",
                          "ok" if bool(en_banda.all()) else "falla",
                          f"tasa {'ponderada' if Bm else 'de la muestra'} 2019-2024: " + ", ".join(f"{a} {ref[a]:.1f} %" for a in range(2019, 2025))
                          + f"; 2025 {ref[2025]:.1f} %. Años fuera de la banda: {[int(a) for a in en_banda.index[~en_banda]] or 'ninguno'}",
                          tabla=_recs(tabla_e)))
    info["tasa_y_por_anio"] = _recs(tabla_e)

    # 16. consistencia de las etiquetas
    ym = base["y_modif"]
    consistente = bool(((base["y_dias"].astype(bool) | ym.astype(bool)).astype("int8") == base["y"]).all()) and not ym.isna().any()
    checks.append(chequeo(16, "y = y_dias | y_modif; y_modif sin faltantes", "ok" if consistente else "falla",
                          f"y {base['y'].dtype}, y_dias {base['y_dias'].dtype}, y_modif {ym.dtype}; faltantes de y_modif {int(ym.isna().sum())}"))

    # 17. madurez por anio y y vs y_dias
    madura = base.groupby("anio_firma")["eval_madura"].mean().reindex(ANIOS) * 100
    tabla_m = pd.DataFrame({"anio": ANIOS, "madura %": madura.round(1).to_numpy(), "excluida (no madura) %": (100 - madura).round(1).to_numpy(),
                            "y muestra %": y_cruda.round(2).to_numpy(), "y_dias %": cruda.round(2).to_numpy(),
                            "y - y_dias pp": (y_cruda - cruda).round(2).to_numpy()})
    monotona = bool((madura.diff().dropna() <= 2.0).all())
    checks.append(chequeo(17, "Fracción madura por año (no crece con el año) y y frente a y_dias por año", "ok" if monotona else "falla",
                          "madura 2019 " + f"{madura[2019]:.1f} %, 2022 {madura[2022]:.1f} %, 2025 {madura[2025]:.1f} %; y supera a y_dias en "
                          f"{(y_cruda - cruda).min():.1f}-{(y_cruda - cruda).max():.1f} pp según el año", tabla=_recs(tabla_m)))

    # 18. AUC univariado
    mask = base["eval_madura"].to_numpy()
    mask = mask if mask.sum() > 100 and len(np.unique(y[mask])) == 2 else np.ones(len(base), dtype=bool)
    auc = auc_univariado(X[mask], y[mask])
    alto = auc[auc > AUC_MAX]
    checks.append(chequeo(18, "Ninguna feature con ROC-AUC univariado > 0,85 sin revisión", "ok" if alto.empty else "falla",
                          f"mayor ROC-AUC univariado: {auc.index[0]} {auc.iloc[0]:.3f}; sobre 0,85: {list(alto.index) or 'ninguna'} "
                          f"({int(mask.sum()):,} contratos maduros; categóricas con codificación fuera de muestra)",
                          tabla=_recs(auc.head(8).round(4).rename("ROC-AUC").reset_index().rename(columns={"index": "feature"}))))
    info["auc_univariado"] = {k: round(float(v), 4) for k, v in auc.items()}

    # 19-21. descarga, modificaciones, sha256
    checks.append(chequeo(19, "La descarga terminó completa (completitud.json ok)", "ok" if ctx["completitud_ok"] else "falla",
                          f"completitud.json ok = {ctx['completitud_ok']}, motivo {ctx.get('completitud_motivo')}"))
    m = ctx["modificaciones"]
    a = ctx["aportes_etiqueta"]
    pos = m.get("fecha_version_posterior_al_corte", 0)
    checks.append(chequeo(20, "Modificaciones: ninguna posterior al corte ni de contratos fuera de la base", "ok" if pos == 0 and a["modificaciones_de_contratos_fuera_de_la_base"] == 0 else "falla",
                          f"{m['filas_leidas']:,} filas leídas, {m['duplicados_exactos']:,} duplicadas exactas ({m['pct_duplicados_exactos']} %), "
                          f"{m['filas_efectivas']:,} efectivas (Publicado, días > 0); con fecha_version posterior al corte {pos} "
                          f"(corte aplicado: {m['corte_aplicado']}); de contratos fuera de la base {a['modificaciones_de_contratos_fuera_de_la_base']}; "
                          f"efectivas sin fecha de aprobación {m['efectivas_sin_fecha_de_aprobacion']}"))
    arch = ctx["archivo"]
    checks.append(chequeo(21, "sha256 del archivo de la base calculado y verificado", "ok" if arch["relectura_igual_a_memoria"] and arch["sha256_verificado_dos_veces"] else "falla",
                          f"{arch['sha256']} ({arch['bytes'] / 2 ** 20:.1f} MB); la relectura del parquet es igual a la base en memoria y el hash es el mismo en dos lecturas"))

    # 22. pesos
    if Bm:
        suma_w = pd.Series(w, index=anio.index).groupby(anio).sum().reindex(ANIOS)
        d = (suma_w / pob_n - 1).abs()
        ok = bool(np.isfinite(w).all() and (w > 0).all() and (d <= 0.01).all())
        checks.append(chequeo(22, "peso_mes = N_mes / n_mes: positivo, finito y su suma por año iguala la población anual", "ok" if ok else "falla",
                              f"fuente de N_mes: {ctx['pesos']['fuente_N_mes']}; pesos entre {w.min():.2f} y {w.max():.2f}; desvío máximo de la suma por año "
                              f"frente a la población {100 * d.max():.2f} %; error esperado de N_mes (y de peso_mes): "
                              + _error_texto(ctx["pesos"].get("error_esperado"))))
    else:
        ok = bool((w == 1.0).all())
        checks.append(chequeo(22, "peso_mes = 1,0 constante (base autoponderada)", "ok" if ok else "falla",
                              f"peso_mes mínimo {w.min()}, máximo {w.max()}; N_mes {'lleno con el conteo exacto' if base['N_mes'].notna().any() else 'vacío (no hay conteo mensual exacto)'}"))

    # ---- informacion (no son chequeos)
    d_pos = {k: int(v) for k, v in a.items()}
    info["positivos_por_parte"] = {**d_pos, "pct_solo_y_dias": _pct(a["solo_y_dias"] / max(a["positivos_y"], 1)),
                                   "pct_solo_y_modif": _pct(a["solo_y_modif"] / max(a["positivos_y"], 1)),
                                   "pct_ambas": _pct(a["ambas"] / max(a["positivos_y"], 1)),
                                   "y_dias_captura_pct_de_y": _pct(a["positivos_y_dias"] / max(a["positivos_y"], 1))}
    n_ = por_anio.to_numpy("float64")
    pos_y = base.groupby("anio_firma")["y"].sum().reindex(ANIOS).to_numpy("float64")
    pos_yd = base.groupby("anio_firma")["y_dias"].sum().reindex(ANIOS).to_numpy("float64")
    pos_pob_est = pd.Series(w * y, index=anio.index).groupby(anio).sum().reindex(ANIOS).to_numpy("float64")
    pos_pob_yd = pob["con_dias_adicionados"].reindex(ANIOS).to_numpy("float64")
    dist = pd.DataFrame({
        "anio": ANIOS, "contratos muestra": n_.astype(int), "% muestra": (100 * n_ / n_.sum()).round(2),
        "contratos población": pob_n.to_numpy().astype(int), "% población": (100 * pob_n / pob_n.sum()).round(2).to_numpy(),
        "razón muestra/población": ((n_ / n_.sum()) / (pob_n.to_numpy() / pob_n.sum())).round(2),
        "positivos y muestra": pos_y.astype(int), "% de positivos muestra": (100 * pos_y / pos_y.sum()).round(2),
        "positivos y población (estimado con pesos)": pos_pob_est.round(0).astype(int),
        "% de positivos población": (100 * pos_pob_est / pos_pob_est.sum()).round(2),
        "positivos y_dias muestra": pos_yd.astype(int), "positivos y_dias población (exacto)": pos_pob_yd.astype(int)})
    info["distribucion_por_anio"] = _recs(dist)
    kish = {int(k): round(P.kish_efectivo(g), 1) for k, g in pd.Series(w, index=anio.index).groupby(anio)}
    info["metricas_ponderadas"] = {
        "peso_mes": {"min": round(float(w.min()), 3), "mediana": round(float(np.median(w)), 3), "max": round(float(w.max()), 3),
                     "suma": round(float(w.sum()), 1)},
        "tamano_efectivo_kish_global": round(P.kish_efectivo(w), 1), "eficiencia_kish_global": round(P.kish_efectivo(w) / len(w), 3),
        "tamano_efectivo_kish_por_anio": kish,
        "tasa_y_ponderada_2019_2025_pct": _pct(np.average(y, weights=w)), "tasa_y_cruda_2019_2025_pct": _pct(y.mean()),
        "tasa_y_dias_ponderada_2019_2025_pct": _pct(np.average(y_dias, weights=w)), "tasa_y_dias_cruda_2019_2025_pct": _pct(y_dias.mean()),
        "uso": "sample_weight=peso_mes en las métricas de evaluación (en A el peso es constante y da lo mismo que la métrica cruda)"}
    if Bm:
        ref_t = pd.DataFrame({"anio": ANIOS, "y_dias muestra % (base)": cruda.round(2).to_numpy(),
                              "referencia §3.7": [REF_Y_DIAS_MUESTRA_PCT[a_] for a_ in ANIOS],
                              "y_dias ponderada % (base)": ponde.round(2).to_numpy(), "referencia §3.7 post-estratificada": [REF_Y_DIAS_POND_PCT[a_] for a_ in ANIOS],
                              "dif vs población pp (base)": (cruda - pob_tasa).round(2).to_numpy(), "referencia §3.7 pp": [REF_DIF_CRUDA_PP[a_] for a_ in ANIOS]})
        info["tasas_vs_seccion_3_7"] = _recs(ref_t)
        info["reproduce_seccion_3_7"] = bool(((ref_t["y_dias muestra % (base)"] - ref_t["referencia §3.7"]).abs() <= 0.05).all())
        mod = base["modalidad_de_contratacion"].fillna("(sin dato)")
        comp = []
        for nom, (r_c, r_p) in REF_COMPOSICION.items():
            comp.append({"modalidad": nom, "cruda %": _pct(np.average(mod == nom)), "ponderada %": _pct(np.average(mod == nom, weights=w)),
                         "referencia §3.8 cruda": r_c, "referencia §3.8 ponderada": r_p})
        info["composicion_modalidad"] = comp
        dire = []
        for a_, (r_c, r_p) in REF_DIRECTA_ANIO.items():
            mk = (anio == a_).to_numpy()
            dire.append({"anio": a_, "directa cruda %": _pct(np.average(mod[mk] == "Contratación directa")),
                         "directa ponderada %": _pct(np.average(mod[mk] == "Contratación directa", weights=w[mk])),
                         "referencia §3.8 cruda": r_c, "referencia §3.8 ponderada": r_p})
        info["contratacion_directa_por_anio"] = dire
    info["pesos_error_esperado"] = ctx["pesos"].get("error_esperado")

    n_falla = sum(c["estado"] == "falla" for c in checks)
    n_bloq = sum(c["estado"] == "falla" and c["bloquea"] for c in checks)
    no_bloq = [f"{c['id']} ({c['chequeo']})" for c in checks if c["estado"] == "falla" and not c["bloquea"]]
    veredicto = (f"NO APTA: {n_bloq} falla(s) bloqueante(s)" if n_bloq else "APTA: sin fallas bloqueantes")
    if no_bloq:
        veredicto += f"; {len(no_bloq)} falla(s) NO bloqueante(s) que el usuario debe revisar: " + "; ".join(no_bloq)
    resumen = {"modo": modo, "chequeos": len(checks), "ok": sum(c["estado"] == "ok" for c in checks), "falla": n_falla,
               "no_aplica": sum(c["estado"] == "no aplica" for c in checks), "fallas_bloqueantes": n_bloq,
               "fallas_no_bloqueantes": len(no_bloq), "veredicto": veredicto}
    return {"version": "v1", "generado": datetime.now().isoformat(timespec="seconds"), "modo": modo, "corte": ctx["corte"],
            "sha256": arch["sha256"], "filas": int(len(base)), "resumen": resumen, "checks": checks, "informacion": info}


# ---------------------------------------------------------------- reporte
def _celda(col, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, bool):
        return "sí" if v else "no"
    if isinstance(v, int):
        return str(v) if col in ("anio", "año") else _es(f"{v:,}")
    if isinstance(v, float):
        return _es(f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.1f}")
    return str(v)


def _md_tabla(recs):
    if not recs:
        return ""
    cols = list(recs[0].keys())
    filas = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    filas += ["| " + " | ".join(_celda(c, r[c]) for c in cols) + " |" for r in recs]
    return "\n".join(filas)


def a_markdown(r, etapas=None):
    s = r["resumen"]
    L = [f"# Reporte de calidad de la base v1, modo {r['modo']}", "",
         _es(f"- Generado: {r['generado']}. Corte de la descarga: {r['corte']}. Filas: {r['filas']:,}. sha256 del parquet: `{r['sha256']}`."),
         f"- **Veredicto: {s['veredicto']}.** Chequeos: {s['chequeos']} ({s['ok']} ok, {s['falla']} falla, {s['no_aplica']} no aplica)."]
    if r["modo"] == "B":
        L.append("- **La base B no es autoponderada**: la muestra es estratificada por mes con 2.000 contratos por mes. Las métricas de evaluación van "
                 "ponderadas con `peso_mes` (y también sin pesos); el entrenamiento es sin pesos.")
    L += ["", "## Resumen de chequeos", "", "| # | Chequeo | Estado | Bloquea |", "|---|---|---|---|"]
    for c in r["checks"]:
        marca = {"ok": "ok", "falla": "**FALLA**", "no aplica": "no aplica"}[c["estado"]]
        L.append(f"| {c['id']} | {c['chequeo']} | {marca} | {'sí' if c['bloquea'] else 'no'} |")
    L += ["", "## Detalle y evidencia", ""]
    for c in r["checks"]:
        L.append(f"### {c['id']}. {c['chequeo']}: {c['estado'].upper()}")
        L += ["", c["evidencia"] + ".", ""]
        if c.get("nota"):
            L += [f"> {c['nota']}", ""]
        if c.get("tabla"):
            L += [_md_tabla(c["tabla"]), ""]
    i = r["informacion"]
    p = i["positivos_por_parte"]
    L += ["## Positivos que aporta cada parte de la definición", "",
          _es(f"- Contratos positivos (y = 1): {p['positivos_y']:,}. Solo por `dias_adicionados > 0` (y_dias): {p['solo_y_dias']:,} ({p['pct_solo_y_dias']} %). "
              f"Solo por Modificaciones (y_modif): {p['solo_y_modif']:,} ({p['pct_solo_y_modif']} %). Por las dos: {p['ambas']:,} ({p['pct_ambas']} %)."),
          _es(f"- `y_dias` captura el {p['y_dias_captura_pct_de_y']} % de los positivos de `y`. Contratos con una extensión publicada: "
              f"{p['contratos_con_extension_publicada']:,}; {p['con_extension_pero_no_cumple_margen']:,} de ellos no cumplen "
              "`fecha_de_aprobacion > fecha_de_firma + 1 día`."), ""]
    L += ["## Tasa de la etiqueta por año", "", _md_tabla(i["tasa_y_por_anio"]), ""]
    L += ["## Distribución de contratos y de positivos por año (muestra frente a población)", "", _md_tabla(i["distribucion_por_anio"]), ""]
    if r["modo"] == "B":
        L += ["### Tasas de y_dias frente a muestra_exploracion.md §3.7", "", _md_tabla(i["tasas_vs_seccion_3_7"]), "",
              f"La base reproduce la muestra cruda de §3.7 (±0,05 pp en los 7 años): {'sí' if i['reproduce_seccion_3_7'] else 'NO'}.", "",
              "### Composición por modalidad, 2019-2025 (§3.8)", "", _md_tabla(i["composicion_modalidad"]), "",
              "### Contratación directa por año (§3.8)", "", _md_tabla(i["contratacion_directa_por_anio"]), ""]
    m = i["metricas_ponderadas"]
    kish = ", ".join(f"{a} {v:,.0f}" for a, v in m["tamano_efectivo_kish_por_anio"].items())
    L += ["## Métricas ponderadas disponibles", "",
          _es(f"- `peso_mes`: mínimo {m['peso_mes']['min']}, mediana {m['peso_mes']['mediana']}, máximo {m['peso_mes']['max']}; suma {m['peso_mes']['suma']:,}."),
          _es(f"- Tamaño efectivo de Kish: {m['tamano_efectivo_kish_global']:,} (eficiencia {m['eficiencia_kish_global']}); por año: {kish}."),
          _es(f"- Tasa de y 2019-2025: cruda {m['tasa_y_cruda_2019_2025_pct']} %, ponderada {m['tasa_y_ponderada_2019_2025_pct']} %. Tasa de y_dias: cruda "
              f"{m['tasa_y_dias_cruda_2019_2025_pct']} %, ponderada {m['tasa_y_dias_ponderada_2019_2025_pct']} %."),
          f"- Uso: {m['uso']}.", ""]
    if r["modo"] == "B":
        L += [_es(f"- Error esperado de `peso_mes`: {_error_texto(i['pesos_error_esperado'])}"), ""]
    if etapas:
        L += ["## Rendimiento de esta corrida", "", "| Etapa | Segundos | RAM pico (MB) |", "|---|---|---|"]
        L += [_es(f"| {k} | {v.get('segundos')} | {v.get('ram_pico_mb')} |") for k, v in etapas.items()]
        L.append("")
    return "\n".join(L)


def escribir(r, carpeta, etapas=None):
    carpeta = Path(carpeta)
    (carpeta / "reporte_calidad.md").write_text(a_markdown(r, etapas), encoding="utf-8")
    (carpeta / "reporte_calidad.json").write_text(json.dumps(r, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
