"""J3 extra: ablacion de `valor_adjudicado_max` y `n_filas` (variables de Procesos que la v1 no usa).

Misma evaluacion y mismo protocolo de src/evaluar_j3_extra.py (HGB final v1, train 2019-2023 maduros, test 2024,
IC 95 % con bootstrap por entidad), pero agregando al HGB final:
  - log1p(valor_adjudicado_max),
  - n_filas (filas del proceso elegido, una por lote o adjudicacion),
  - ambas.
Las dos variables NO estan en la base v1 (se excluyeron por ambiguedad de la medida, no por fuga): se calculan aqui con
elegir_proceso (src/uniones.py) sobre data/raw/f2/procesos y se unen en memoria por proceso_de_compra. No se guardan en
la base ni entran en FEATURES. La referencia de las diferencias es el HGB final (no la regla por entidad).

Uso (desde final-project/):  python -m src.evaluar_j3_valor_adjudicado
Salida: resultados/j3_extra_valor_adjudicado.json y .md
"""
import datetime
import glob
import json
import time

import numpy as np
import pandas as pd

from . import config
from . import evaluacion as E
from . import modelado as MD
from . import modelo as MO
from .evaluar_j3_extra import N_BOOT, RES, _pipeline_con_columnas
from .uniones import elegir_proceso

RUTA_PROCESOS = config.DATA_RAW / "f2" / "procesos"
COLS_PROCESOS = ["id_del_portafolio", "id_del_proceso", "adjudicado", "fecha_adjudicacion",
                 "valor_total_adjudicacion"]
EXTRAS_VALOR = ["log_valor_adjudicado_max"]
EXTRAS_N = ["n_filas"]
METRICAS_RESUMEN = ("pr_auc", "roc_auc")
VARIANTES = {"final": [], "final_mas_valor": EXTRAS_VALOR, "final_mas_n_filas": EXTRAS_N,
             "final_mas_ambos": EXTRAS_VALOR + EXTRAS_N}
ETIQUETA = {"final": "final (39 features)", "final_mas_valor": "+ log1p(valor_adjudicado_max)",
            "final_mas_n_filas": "+ n_filas", "final_mas_ambos": "+ ambas"}


def leer_procesos():
    """Lee solo COLS_PROCESOS del crudo de Procesos y deja una fila por portafolio con elegir_proceso."""
    archivos = sorted(glob.glob(str(RUTA_PROCESOS / "lote_*.parquet")))
    if not archivos:
        raise FileNotFoundError(f"no hay crudo de Procesos en {RUTA_PROCESOS}")
    procesos = pd.concat([pd.read_parquet(f, columns=COLS_PROCESOS) for f in archivos], ignore_index=True)
    return procesos, elegir_proceso(procesos)


def unir_extras(base, elegido):
    """Devuelve las columnas extra alineadas al indice de `base` (NaN si el contrato no esta en Procesos)."""
    p = elegido.set_index("id_del_portafolio")
    llave = base["proceso_de_compra"]
    valor = llave.map(p["valor_adjudicado_max"])
    extras = pd.DataFrame({"log_valor_adjudicado_max": np.log1p(valor.clip(lower=0)),
                           "n_filas": llave.map(p["n_filas"]).astype("float64")}, index=base.index)
    fecha_adj = pd.to_datetime(llave.map(p["fecha_adjudicacion"]), errors="coerce", format="ISO8601").dt.normalize()
    return extras, valor, fecha_adj


def calidad(base, procesos, elegido, valor, fecha_adj):
    """Cobertura del join, ceros y orden de la fecha de adjudicacion respecto a la firma."""
    firma = pd.to_datetime(base["fecha_de_firma"], errors="coerce", format="ISO8601").dt.normalize()
    emparejados = base["eval_en_procesos"].to_numpy()
    dias = (fecha_adj - firma).dt.days.dropna()
    return {"contratos_base": int(len(base)), "portafolios": int(len(elegido)), "filas_procesos": int(len(procesos)),
            "cobertura": round(float(emparejados.mean()), 4),
            "valor_cero": round(float((valor[emparejados] == 0).mean()), 4),
            "contratos_valor_positivo": int((valor > 0).sum()),
            "portafolios_n_filas_mayor_1": int((elegido["n_filas"] > 1).sum()),
            "contratos_con_fecha_adjudicacion": int(len(dias)),
            "pct_con_fecha_adjudicacion": round(len(dias) / len(base), 4),
            "pct_adjudicacion_mismo_dia_o_antes_firma": round(float((dias <= 0).mean()), 5),
            "pct_adjudicacion_despues_firma": round(float((dias > 0).mean()), 5),
            "mediana_dias_adjudicacion_a_firma": float(dias.median()),
            "p1_dias_adjudicacion_a_firma": float(dias.quantile(0.01)),
            "p99_dias_adjudicacion_a_firma": float(dias.quantile(0.99))}


def main(log=print):
    RES.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    base, X, idx_tr = MD.cargar()
    _, idx_te = E.split_temporal(base, config.ANIO_TEST)
    y_tr, y_te = base.loc[idx_tr, "y"].to_numpy(), base.loc[idx_te, "y"].to_numpy()
    ent_te = base.loc[idx_te, "nit_entidad"].to_numpy()

    procesos, elegido = leer_procesos()
    extras, valor, fecha_adj = unir_extras(base, elegido)
    q = calidad(base, procesos, elegido, valor, fecha_adj)
    log(f"procesos leidos y unidos ({time.perf_counter() - t0:.0f} s): {q['filas_procesos']} filas, "
        f"{q['portafolios']} portafolios, cobertura {q['cobertura']:.4f}")
    Xe = pd.concat([X, extras], axis=1)

    scores = {}
    for nombre, cols in VARIANTES.items():
        if cols:
            pipe = _pipeline_con_columnas([("extra", "passthrough", cols)])
        else:
            pipe = MO.construir_pipeline()
        entrada = Xe if cols else X
        scores[nombre] = pipe.fit(entrada.loc[idx_tr], y_tr).predict_proba(entrada.loc[idx_te])[:, 1]
        log(f"  {nombre} listo ({time.perf_counter() - t0:.0f} s)")

    t = E.comparar(y_te, scores, ent_te, referencia="final", n_boot=N_BOOT)

    def fila(modelo, metrica):
        return t[(t["modelo"] == modelo) & (t["metrica"] == metrica)].iloc[0]

    out = {"protocolo": "HGB final v1, train 2019-2023, test 2024, solo maduros; igual que src/evaluar_j3_extra.py",
           "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)), "tasa_base_test": round(float(y_te.mean()), 4),
           "cobertura": q["cobertura"], "valor_cero": q["valor_cero"], "n_procesos": q["filas_procesos"],
           "n_portafolios": q["portafolios"],
           "punto": {k: {m: round(float(fila(k, m)["valor"]), 6) for m in METRICAS_RESUMEN} for k in scores},
           "deltas_vs_final": {k: {m: round(float(fila(k, m)["dif"]), 6) for m in METRICAS_RESUMEN}
                               for k in scores if k != "final"},
           "segundos": round(time.perf_counter() - t0),
           "fecha_corrida": datetime.date.today().isoformat(),
           "script": "src/evaluar_j3_valor_adjudicado.py; mismo pipeline que src/evaluar_j3_extra.py",
           "calidad": q,
           "tabla": json.loads(t.to_json(orient="records"))}
    (RES / "j3_extra_valor_adjudicado.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float),
                                                         encoding="utf-8")
    (RES / "j3_extra_valor_adjudicado.md").write_text(a_markdown(out, t), encoding="utf-8")
    log(f"listo en {out['segundos']} s")
    return out


def _n(x, dec=4):
    return f"{x:.{dec}f}".replace(".", ",")


def _miles(x):
    return f"{int(x):,}".replace(",", ".")


def _pct(x, dec=2):
    return f"{100 * x:.{dec}f}".replace(".", ",") + " %"


def _signo(x, dec=4):
    return f"{'+' if x >= 0 else '-'}{_n(abs(x), dec)}".replace("-", "−")


def a_markdown(r, t):
    q = r["calidad"]
    d = datetime.date.fromisoformat(r["fecha_corrida"])
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

    def fila(modelo, metrica):
        return t[(t["modelo"] == modelo) & (t["metrica"] == metrica)].iloc[0]

    def dif(modelo, metrica):
        f = fila(modelo, metrica)
        txt = f"{_signo(f['dif'])} [{_signo(f['dif_ic_inf'])}; {_signo(f['dif_ic_sup'])}]"
        return txt.replace("−", "-")

    filas = []
    for k, nombre in ETIQUETA.items():
        pr, roc = r["punto"][k]["pr_auc"], r["punto"][k]["roc_auc"]
        if k == "final":
            filas.append(f"| {nombre} | {_n(pr)} | {_n(roc)} | — | — |")
        else:
            filas.append(f"| {nombre} | {_n(pr)} | {_n(roc)} | {dif(k, 'pr_auc')} | {dif(k, 'roc_auc')} |")
    incluyen_cero = all(t[(t["modelo"] != "final") & (t["metrica"].isin(METRICAS_RESUMEN))]
                        .apply(lambda f: f["dif_ic_inf"] <= 0 <= f["dif_ic_sup"], axis=1))
    L = ["# J3 extra: ablación de `valor_total_adjudicacion` / `valor_adjudicado_max` / `n_filas`", "",
         f"Corrida del {d.day}-{meses[d.month - 1]}-{d.year} con el mismo protocolo de `src/evaluar_j3_extra.py`: "
         f"HGB final v1, train 2019-2023, test 2024, solo maduros. Reproduce el `final` de "
         f"`resultados/j3_extra_resumen.json` (PR-AUC {_n(r['punto']['final']['pr_auc'])}; ROC-AUC "
         f"{_n(r['punto']['final']['roc_auc'])}), así que el protocolo es comparable. "
         "Script: `src/evaluar_j3_valor_adjudicado.py`.", "",
         "## Por qué se excluyó de la v1", "",
         "- `00_exploracion/exploracion_secop.ipynb` §4b y §8: con varios lotes/proveedores el valor adjudicado no "
         "es el valor del contrato; `valor_adjudicado_max` y `n_filas` quedan fuera por **ambigüedad de la "
         "medida**, no por fuga.",
         f"- El {_pct(q['valor_cero'])} de los contratos tiene `valor_adjudicado_max = 0`; solo "
         f"{_miles(q['contratos_valor_positivo'])} tienen valor positivo. La columna es casi un indicador de "
         "\"hubo adjudicación formal\", y ese indicador no discrimina.",
         f"- De los {_miles(q['contratos_con_fecha_adjudicacion'])} contratos con `fecha_adjudicacion` "
         f"({_pct(q['pct_con_fecha_adjudicacion'])} de la base), el {_pct(q['pct_adjudicacion_mismo_dia_o_antes_firma'])} "
         f"se adjudicó el mismo día o antes de la firma y el {_pct(q['pct_adjudicacion_despues_firma'])} después. "
         "La columna se conoce en la adjudicación, antes de la firma en la enorme mayoría; no hay evidencia de fuga "
         "temporal, pero tampoco se puede verificar con una sola foto.", "",
         "## Resultado", "",
         "| Variante | PR-AUC | ROC-AUC | ΔPR-AUC vs final [IC 95 %] | ΔROC-AUC vs final [IC 95 %] |",
         "|---|---:|---:|---|---|", *filas, "",
         f"IC 95 % con bootstrap por entidad ({N_BOOT} repeticiones). "
         + ("Todas las diferencias incluyen 0. Agregar la columna no aporta señal; se mantiene fuera de `FEATURES`."
            if incluyen_cero else "**Alguna diferencia no incluye 0: revisar antes de mantener la columna fuera.**"),
         "", "## Calidad de la columna", "",
         f"- Cobertura del join con la base: {_pct(q['cobertura'])} ({_miles(q['portafolios'])} portafolios; "
         f"{_miles(q['filas_procesos'])} filas de Procesos).",
         f"- `valor_adjudicado_max = 0`: {_pct(q['valor_cero'])} de los contratos.",
         f"- `n_filas > 1`: {_pct(q['portafolios_n_filas_mayor_1'] / q['portafolios'])} de los portafolios "
         f"({_miles(q['portafolios_n_filas_mayor_1'])} de {_miles(q['portafolios'])}).",
         f"- Fechas: {_miles(q['contratos_con_fecha_adjudicacion'])} contratos con `fecha_adjudicacion` "
         f"({_pct(q['pct_con_fecha_adjudicacion'])}); mediana {_signo(q['mediana_dias_adjudicacion_a_firma'], 0)} "
         f"días respecto a la firma; p1 {_signo(q['p1_dias_adjudicacion_a_firma'], 0)} días; p99 "
         f"{_signo(q['p99_dias_adjudicacion_a_firma'], 0).replace('+', '')} días.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
