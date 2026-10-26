"""Pruebas del preprocesamiento (src/preprocesamiento.py). Corren sin datos: features sinteticas."""
import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src import features as F
from src import preprocesamiento as P


def _X(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(index=pd.RangeIndex(n))
    for c in F.FEATURES_CAT:
        X[c] = rng.choice(["a", "b", "c", np.nan], n, p=[.5, .3, .15, .05]).astype(object)
    X["nit_entidad"] = rng.choice([f"9000{i}" for i in range(15)], n).astype(object)
    for c in F.FEATURES_NUM:
        X[c] = rng.integers(0, 50, n).astype("float64")
    X["mes_firma"] = rng.integers(1, 13, n).astype("float64")
    X["dia_semana"] = rng.integers(0, 7, n).astype("float64")
    X["p_dur_dias"] = rng.choice([0.0, 30.0, 120.0, 365.0, np.nan], n)
    X.loc[0, "p_dur_dias"] = 4.078e9          # el atipico de la v1
    X["p_proveedores_que_manifestaron"] = 0.0
    X = X.where(X.notna(), np.nan)
    y = (rng.random(n) < 0.3).astype(int)
    return X[F.FEATURES], y


def _nombres_validos(nombres):
    """Misma regla que src/explicacion.py (_original_de): cada salida es 'feature' o 'feature_algo'."""
    for n in nombres:
        n = n.split("__")[-1]
        assert any(n == f or n.startswith(f + "_") for f in F.FEATURES), n


@pytest.mark.parametrize("familia", P.FAMILIAS)
def test_familias_sin_fallar_con_categorias_nuevas_y_nombres_mapeables(familia):
    X, y = _X()
    ct = P.construir_preprocesador(familia).fit(X, y)
    nuevo = X.head(5).copy()
    nuevo["tipo_de_contrato"] = "categoria nunca vista"
    nuevo["nit_entidad"] = "entidad nueva"
    Z = ct.transform(nuevo)
    assert len(Z) == 5
    _nombres_validos(ct.get_feature_names_out())
    assert "p_proveedores_que_manifestaron" not in Z.columns       # constante, se descarta
    if familia == "distancia":
        assert not Z.isna().any().any()


def test_duracion_cero_a_faltante_y_winsorizada_con_cuantil_del_train():
    X, y = _X()
    ct = P.construir_preprocesador("arbol").fit(X, y)
    d = ct.named_transformers_["dur"]
    positivos = X["p_dur_dias"][X["p_dur_dias"] > 0]
    assert d.tope_[0] == pytest.approx(positivos.quantile(P.CUANTIL_DURACION))
    Z = ct.transform(X)
    assert Z["p_dur_dias"].max() <= d.tope_[0]
    assert Z.loc[X["p_dur_dias"] == 0, "p_dur_dias"].isna().all()
    assert (Z["p_dur_dias_cero"] == (X["p_dur_dias"] == 0)).all()


def test_ajuste_solo_con_train():
    """Lo aprendido (tope, codigos) depende solo de las filas de fit: cambiar el test no lo mueve."""
    X, y = _X()
    a = P.construir_preprocesador("arbol").fit(X.iloc[:200], y[:200])
    X2 = X.copy()
    X2.iloc[200:, X2.columns.get_loc("p_dur_dias")] = 9999.0
    b = P.construir_preprocesador("arbol").fit(X2.iloc[:200], y[:200])
    assert a.named_transformers_["dur"].tope_ == b.named_transformers_["dur"].tope_


def test_excluir_para_la_ablacion():
    X, y = _X()
    ct = P.construir_preprocesador("arbol", excluir=F.FEATURES_ABLACION_J3).fit(X, y)
    assert not set(F.FEATURES_ABLACION_J3) & set(ct.get_feature_names_out())
    with pytest.raises(ValueError):
        P.construir_preprocesador("arbol", excluir=["no_existe"])


def test_hgb_arbol_sin_nativas_y_shap_aditivo():
    X, y = _X()
    pipe = P.pipeline_hgb_defecto().fit(X, y)
    assert not np.any(pipe[-1].is_categorical_)        # condicion de la app (src/explicacion.py)
    Xt = pipe[:-1].transform(X.head(20))
    ex = shap.TreeExplainer(pipe[-1])
    v = ex.shap_values(Xt)
    err = np.abs(v.sum(1) + np.ravel(ex.expected_value)[0] - pipe[-1].decision_function(Xt)).max()
    assert err < 1e-3


def test_variante_nativa_si_declara_categorias():
    X, y = _X()
    hgb = HistGradientBoostingClassifier(early_stopping=False, categorical_features="from_dtype", max_iter=10)
    pipe = P.pipeline(hgb, "arbol_nativo").fit(X, y)
    assert pipe[-1].is_categorical_.sum() == len(F.FEATURES_CAT) - 1   # todas menos nit_entidad (TargetEncoder)


def test_distancia_alimenta_un_modelo_lineal():
    X, y = _X()
    pipe = P.pipeline(LogisticRegression(max_iter=500), "distancia").fit(X, y)
    assert pipe.predict_proba(X).shape == (len(X), 2)
