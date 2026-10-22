"""Modelo dummy para que Geronimo construya la app mientras llega el modelo real (contrato 2d).

HGB con los hiperparametros por defecto (sin early stopping: la regla 4 exige validacion temporal o desactivarlo)
sobre la base v0, entrenado solo con los anios de train (2019-2023) y contratos maduros. Las categorias van como
codigos numericos dentro del Pipeline (sin categorias nativas), asi TreeExplainer de SHAP es exacto (plan, seccion 0,
punto 7). NO es el modelo de produccion: sus cifras no se citan.
Uso (desde final-project/, con la base v0 ya construida):  python -m src.modelo_dummy
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from . import base as B
from . import config
from . import features as F
from . import modelo_io

RUTA_DUMMY = config.MODELS / "modelo_prorroga_dummy.joblib"
TOP_CATEGORIAS = 200   # como la exploracion: las 200 mas frecuentes; el resto, un solo codigo


def construir_pipeline():
    """Codigos ordinales para las categoricas (desconocidas y faltantes -> NaN) + HGB por defecto."""
    codigos = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan,
                             max_categories=TOP_CATEGORIAS)
    prep = ColumnTransformer([("cat", codigos, F.FEATURES_CAT)], remainder="passthrough", verbose_feature_names_out=False)
    prep.set_output(transform="pandas")
    return Pipeline([("prep", prep), ("hgb", HistGradientBoostingClassifier(early_stopping=False, random_state=config.SEED))])


def entrenar_dummy(base):
    """Entrena con 2019-2023 (maduros), mide en 2024 como verificacion de cordura y guarda el artefacto."""
    train = base[(base["anio_firma"] >= config.ANIO_INICIO) & (base["anio_firma"] < config.ANIO_TEST) & base["eval_madura"]]
    test = base[(base["anio_firma"] == config.ANIO_TEST) & base["eval_madura"]]
    pipe = construir_pipeline().fit(F.construir_features(train), train["y"])

    puntaje = pipe.predict_proba(F.construir_features(test))[:, 1]
    roc, pr = roc_auc_score(test["y"], puntaje), average_precision_score(test["y"], puntaje)
    # Referencia para que la app calcule el percentil de un contrato nuevo: cuantiles del puntaje en 2024 (aun no visto).
    cuantiles = np.quantile(puntaje, np.linspace(0, 1, 101))
    meta = {
        "version": "dummy-0",
        "advertencia": "Modelo dummy: solo sirve para probar el flujo de la app. No es el modelo de produccion y no se citan sus cifras.",
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features_version": F.FEATURES_VERSION,
        "features": F.FEATURES,
        "etiqueta": "y = dias_adicionados > 0 (base v0)",
        "datos": {"base": B.BASE_VERSION, "corte": str(B.FECHA_CORTE_V0.date()), "anios_train": [config.ANIO_INICIO, config.ANIO_TEST - 1],
                  "n_train": int(len(train)), "tasa_base_train": round(float(train["y"].mean()), 4),
                  "digito": "no aplica (muestra de la exploracion)"},
        "modelo": {"clase": "HistGradientBoostingClassifier", "hiperparametros": "por defecto, early_stopping=False",
                   "categoricas": f"OrdinalEncoder (max {TOP_CATEGORIAS} categorias, sin categorias nativas)", "semilla": config.SEED},
        "metricas": {"test_2024": {"n": int(len(test)), "tasa_base": round(float(test["y"].mean()), 4),
                                   "roc_auc": round(float(roc), 4), "pr_auc": round(float(pr), 4)}},
        # Provisional: la app muestra nivel y percentil, no la probabilidad cruda. El modelo v1 los fija con validacion temporal.
        "umbrales_riesgo": {"metodo": "cuantiles del puntaje en 2024 (provisional)", "medio": round(float(cuantiles[70]), 6),
                            "alto": round(float(cuantiles[90]), 6)},
        "percentiles_puntaje": [round(float(q), 6) for q in cuantiles],
        "entorno": modelo_io.entorno(),
        "nombres_legibles": {f: F.NOMBRES_LEGIBLES[f] for f in F.FEATURES},
    }
    return pipe, modelo_io.guardar_modelo(pipe, meta, RUTA_DUMMY)


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    _base = B.validar_esquema(pd.read_parquet(config.DATA_PROCESSED / "base_contratos.parquet"))
    _pipe, _meta = entrenar_dummy(_base)
    m = _meta["metricas"]["test_2024"]
    print(f"dummy entrenado en {time.perf_counter() - t0:.0f} s con {_meta['datos']['n_train']} contratos; "
          f"2024 (n={m['n']}, tasa base {m['tasa_base']}): ROC-AUC {m['roc_auc']}, PR-AUC {m['pr_auc']}")
    print("sha256", _meta["sha256"], "| tamano:", round(RUTA_DUMMY.stat().st_size / 1e3), "KB")
