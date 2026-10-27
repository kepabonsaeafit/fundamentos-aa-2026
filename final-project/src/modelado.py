"""J3: busqueda de hiperparametros de los 5 modelos del curso con folds temporales dentro del train 2019-2023.

- cv = folds_temporales (ventana creciente, validacion 2021, 2022 y 2023); el Pipeline completo (con el TargetEncoder)
  se reajusta en cada fold, asi que no hay fuga entre folds (regla 4). 2024 y 2025 no se tocan aqui.
- Metrica de seleccion: PR-AUC medio de los folds (regla 6); tambien se guarda el ROC-AUC.
- Tope de 15 min por modelo (plan, limites de tiempo): n_iter se fijo midiendo un ajuste; kNN y SVM sobre una
  submuestra de 20.000 contratos del train. Si una busqueda pasa el tope se reporta, no se esconde.
- Early stopping desactivado en el HGB (regla 4).

Uso (desde final-project/):  python -m src.modelado [--modelos hgb rf ...]
Salida: resultados/j3_busqueda_<modelo>.csv y resultados/j3_busqueda_resumen.json
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config
from . import evaluacion as E
from . import features as F
from . import preprocesamiento as P

RESULTADOS = config.RAIZ / "resultados"
BASE_V1 = config.DATA_PROCESSED / "v1_A" / "base_contratos.parquet"
TOPE_SEGUNDOS = 15 * 60
N_SUBMUESTRA = 20_000

# nombre -> (estimador, familia de preprocesamiento, espacio, n_iter, submuestra?, n_jobs de la busqueda)
MODELOS = {
    "hgb": (HistGradientBoostingClassifier(early_stopping=False, random_state=config.SEED), "arbol",
            {"modelo__learning_rate": loguniform(0.02, 0.3), "modelo__max_iter": [100, 200, 400],
             "modelo__max_leaf_nodes": [15, 31, 63, 127], "modelo__min_samples_leaf": [20, 50, 100, 200],
             "modelo__l2_regularization": loguniform(1e-3, 10), "modelo__max_features": [0.5, 0.8, 1.0]},
            25, False, 2),
    "arbol": (DecisionTreeClassifier(random_state=config.SEED), "arbol",
              {"modelo__max_depth": [4, 6, 8, 10, 12, 15, 20, None], "modelo__min_samples_leaf": [10, 20, 50, 100, 200, 500],
               "modelo__criterion": ["gini", "entropy"], "modelo__class_weight": [None, "balanced"]},
              30, False, 4),
    "rf": (RandomForestClassifier(n_jobs=-1, random_state=config.SEED), "arbol",
           {"modelo__n_estimators": [200, 400], "modelo__max_depth": [None, 12, 20],
            "modelo__min_samples_leaf": [5, 20, 50], "modelo__max_features": ["sqrt", 0.3, 0.5]},
           12, False, 1),
    "knn": (KNeighborsClassifier(), "distancia",
            {"modelo__n_neighbors": [10, 25, 50, 100, 200, 300], "modelo__weights": ["uniform", "distance"],
             "modelo__p": [1, 2]},
            15, True, 4),
    "svm": (SVC(kernel="rbf", random_state=config.SEED), "distancia",
            {"modelo__C": loguniform(0.1, 10), "modelo__gamma": ["scale", 0.003, 0.01, 0.03]},
            10, True, 4),
}


def cargar():
    """Base v1, features y posiciones del train 2019-2023 (maduros)."""
    base = pd.read_parquet(BASE_V1)
    X = F.construir_features(base)
    tr, _ = E.split_temporal(base, config.ANIO_TEST)
    return base, X, tr


def buscar(nombre, base, X, idx_train, log=print):
    est, familia, espacio, n_iter, submuestra, n_jobs = MODELOS[nombre]
    idx = idx_train
    if submuestra:
        rng = np.random.default_rng(config.SEED)
        idx = idx_train[np.sort(rng.choice(len(idx_train), N_SUBMUESTRA, replace=False))]
    anios = base.loc[idx, "anio_firma"].to_numpy()
    cv = E.folds_temporales(anios)
    busqueda = RandomizedSearchCV(P.pipeline(est, familia), espacio, n_iter=n_iter, cv=cv,
                                  scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"}, refit=False,
                                  n_jobs=n_jobs, random_state=config.SEED, error_score="raise")
    t0 = time.perf_counter()
    busqueda.fit(X.loc[idx], base.loc[idx, "y"].to_numpy())
    seg = time.perf_counter() - t0
    res = pd.DataFrame(busqueda.cv_results_).sort_values("rank_test_pr_auc")
    RESULTADOS.mkdir(exist_ok=True)
    res.drop(columns=["params"]).to_csv(RESULTADOS / f"j3_busqueda_{nombre}.csv", index=False)
    mejor = res.iloc[0]
    out = {"modelo": nombre, "familia": familia, "n_iter": n_iter, "n_train": int(len(idx)), "segundos": round(seg),
           "dentro_del_tope": seg <= TOPE_SEGUNDOS, "pr_auc_cv": float(mejor["mean_test_pr_auc"]),
           "pr_auc_cv_std": float(mejor["std_test_pr_auc"]), "roc_auc_cv": float(mejor["mean_test_roc_auc"]),
           "mejores_parametros": {k.removeprefix("modelo__"): (v.item() if hasattr(v, "item") else v)
                                  for k, v in mejor["params"].items()}}
    log(f"{nombre}: {seg:.0f} s, PR-AUC cv {out['pr_auc_cv']:.4f} ± {out['pr_auc_cv_std']:.4f}, "
        f"ROC-AUC cv {out['roc_auc_cv']:.4f}, {out['mejores_parametros']}")
    return out


def estimador_ajustado(nombre, parametros):
    """Pipeline del modelo `nombre` con los parametros elegidos (sin ajustar)."""
    est, familia, *_ = MODELOS[nombre]
    return P.pipeline(est, familia).set_params(**{f"modelo__{k}": v for k, v in parametros.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=list(MODELOS), choices=list(MODELOS))
    a = ap.parse_args()
    ruta = RESULTADOS / "j3_busqueda_resumen.json"
    resumen = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    _base, _X, _tr = cargar()
    for m in a.modelos:
        resumen[m] = buscar(m, _base, _X, _tr)
        ruta.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
