"""Modelo de produccion v1 (J3, duenio: Jose): construir_pipeline() y el artefacto models/modelo_prorroga_v1.

- construir_pipeline(): Pipeline sin ajustar con el preprocesamiento de J2 (familia "arbol": codigos ordinales,
  TargetEncoder para la entidad, sin categorias nativas) y el HGB con los hiperparametros elegidos en la busqueda
  temporal (resultados/j3_busqueda_resumen.json). Lo usa el monitoreo de G3 para reentrenar campeon y retador.
- entrenar_v1(): entrena con 2019-2023 (maduros), puntua el test 2024 (maduros) y guarda el artefacto con modelo_io.
  Los percentiles del puntaje salen del test 2024 (no del train, donde el puntaje es optimista).

Uso (desde final-project/):  python -m src.modelo
"""
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config
from . import evaluacion as E
from . import features as F
from . import modelo_io
from . import preprocesamiento as P

VERSION = "v1.0"
RUTA_V1 = config.MODELS / "modelo_prorroga_v1.joblib"
BASE_V1 = config.DATA_PROCESSED / "v1_A" / "base_contratos.parquet"
FAMILIA = "arbol"
# Mejor configuracion de la busqueda temporal (src/modelado.py, PR-AUC medio en folds 2021-2023).
HIPERPARAMETROS_V1 = {"l2_regularization": 0.002259279742015696, "learning_rate": 0.034003508264422166,
                      "max_features": 0.5, "max_iter": 200, "max_leaf_nodes": 127, "min_samples_leaf": 20}
# Percentiles del puntaje en 2024 que separan los niveles. PROVISIONAL: se cuadran con Geronimo (app y monitoreo).
PERCENTIL_MEDIO, PERCENTIL_ALTO = 70, 90


def construir_pipeline(excluir=()):
    """Pipeline del modelo v1 sin ajustar. Recibe la salida de construir_features."""
    hgb = HistGradientBoostingClassifier(early_stopping=False, random_state=config.SEED, categorical_features=None,
                                         **HIPERPARAMETROS_V1)
    return P.pipeline(hgb, FAMILIA, excluir)


def metadatos(base, idx_tr, idx_te, puntaje_2024, metricas_extra=None):
    y_tr, y_te = base.loc[idx_tr, "y"], base.loc[idx_te, "y"]
    cuantiles = np.quantile(puntaje_2024, np.linspace(0, 1, 101))
    m = E.metricas(y_te, puntaje_2024)
    return {
        "version": VERSION,
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features_version": F.FEATURES_VERSION,
        "features": F.FEATURES,
        "etiqueta": "y = y_dias | y_modif (ampliacion del plazo; base v1_A, docs/01_definicion_problema.md)",
        "datos": {"base": "v1_A", "corte": "2026-10-26", "digito": 4, "muestra": "10 % por digito 4 del id_contrato",
                  "anios_train": [config.ANIO_INICIO, config.ANIO_TEST - 1], "n_train": int(len(idx_tr)),
                  "tasa_base_train": round(float(y_tr.mean()), 4), "solo_maduros": True},
        "modelo": {"clase": "HistGradientBoostingClassifier", "hiperparametros": {"early_stopping": False, **HIPERPARAMETROS_V1},
                   "preprocesamiento": "src/preprocesamiento.py familia 'arbol': OrdinalEncoder (top 200), TargetEncoder "
                                       "para nit_entidad, p_dur_dias 0 -> NaN y winsorizada al p99,9 del train",
                   "categorias_nativas": False, "semilla": config.SEED},
        "metricas": {"test_2024": {"n": int(len(idx_te)), "tasa_base": round(float(y_te.mean()), 4),
                                   "roc_auc": round(float(m["roc_auc"]), 4), "pr_auc": round(float(m["pr_auc"]), 4),
                                   **{k: round(float(m[k]), 4) for k in ("p@1%", "p@5%", "p@10%")}},
                     **(metricas_extra or {})},
        "umbrales_riesgo": {"metodo": f"percentiles {PERCENTIL_MEDIO} y {PERCENTIL_ALTO} del puntaje en el test 2024 "
                                      "(maduros); provisional, se cuadra con Geronimo",
                            "medio": round(float(cuantiles[PERCENTIL_MEDIO]), 6),
                            "alto": round(float(cuantiles[PERCENTIL_ALTO]), 6)},
        "percentiles_puntaje": [round(float(q), 6) for q in cuantiles],
        "percentiles_fuente": "test 2024, contratos maduros (no el train)",
        "entidades_vistas": sorted(base.loc[idx_tr, "nit_entidad"].dropna().astype(str).unique().tolist()),
        "entorno": modelo_io.entorno(),
        "nombres_legibles": {f: F.NOMBRES_LEGIBLES[f] for f in F.FEATURES},
    }


def entrenar_v1(base=None, metricas_extra=None, ruta=RUTA_V1):
    base = pd.read_parquet(BASE_V1) if base is None else base
    X = F.construir_features(base)
    idx_tr, idx_te = E.split_temporal(base, config.ANIO_TEST)
    pipe = construir_pipeline().fit(X.loc[idx_tr], base.loc[idx_tr, "y"])
    puntaje = pipe.predict_proba(X.loc[idx_te])[:, 1]
    meta = metadatos(base, idx_tr, idx_te, puntaje, metricas_extra)
    return pipe, modelo_io.guardar_modelo(pipe, meta, ruta)


if __name__ == "__main__":
    t0 = time.perf_counter()
    _pipe, _meta = entrenar_v1()
    m = _meta["metricas"]["test_2024"]
    print(f"v1 entrenado en {time.perf_counter() - t0:.0f} s; 2024 (n={m['n']}, tasa {m['tasa_base']}): "
          f"ROC-AUC {m['roc_auc']}, PR-AUC {m['pr_auc']} | {RUTA_V1.stat().st_size / 1e6:.1f} MB")
