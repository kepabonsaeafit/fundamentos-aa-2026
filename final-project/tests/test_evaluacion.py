"""Pruebas minimas del marco de evaluacion (src/evaluacion.py). Corren sin datos (filas sinteticas)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import evaluacion as E


def _base_sintetica(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    anio = rng.integers(2019, 2026, n)
    ent = rng.integers(0, 40, n).astype(str)
    tasa_ent = pd.Series(ent).map({str(i): 0.1 + 0.6 * (i % 7) / 6 for i in range(40)}).to_numpy()
    y = (rng.random(n) < tasa_ent).astype("int8")
    return pd.DataFrame({"anio_firma": anio, "nit_entidad": ent, "y": y,
                         "eval_madura": rng.random(n) > 0.05,
                         "eval_dur_planeada_dias": rng.choice([30, 120, 300, 500, np.nan], n)})


def test_split_temporal_sin_mezclar_anios_y_solo_maduros():
    b = _base_sintetica()
    tr, te = E.split_temporal(b, 2024)
    assert b.loc[tr, "anio_firma"].between(2019, 2023).all()
    assert (b.loc[te, "anio_firma"] == 2024).all()
    assert b.loc[tr.union(te), "eval_madura"].all()
    tr5, te5 = E.split_temporal(b, 2025, anio_fin_train=2023)   # deriva (i)
    assert b.loc[tr5, "anio_firma"].max() == 2023 and (b.loc[te5, "anio_firma"] == 2025).all()
    with pytest.raises(ValueError):
        E.split_temporal(b, 2024, anio_fin_train=2024)


def test_folds_temporales_ventana_creciente():
    anios = np.array([2019, 2020, 2021, 2022, 2023] * 3)
    folds = E.folds_temporales(anios)
    assert len(folds) == 3
    for (tr, va), v in zip(folds, (2021, 2022, 2023)):
        assert (anios[tr] < v).all() and (anios[va] == v).all()


def test_regla_entidad_suavizado_y_no_vistas():
    ent_tr = np.array(["a", "a", "b", "b", "b", "b"])
    y_tr = np.array([1, 1, 0, 0, 0, 1])
    p0 = y_tr.mean()
    s = E.regla_tasa_entidad(ent_tr, y_tr, np.array(["a", "b", "z"]), m=2)
    assert s[0] == pytest.approx((2 + 2 * p0) / (2 + 2))
    assert s[1] == pytest.approx((1 + 2 * p0) / (4 + 2))
    assert s[2] == pytest.approx(p0)
    assert np.allclose(E.regla_tasa_base(y_tr, 3), p0)


def test_metricas_top_k_con_empates_y_recall():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([.9, .8, .8, .1, .1, .1, .1, .1, .1, .1])
    m = E.metricas(y, s, fracciones_k=(0.1, 0.3))
    assert m["p@10%"] == pytest.approx(1.0)                # k=1: el .9 es positivo
    assert m["p@30%"] == pytest.approx((1 + 2 * 0.5) / 3)  # k=3: .9 + los dos empatados en .8 (uno positivo)
    assert m["r@30%"] == pytest.approx(m["p@30%"] * 3 / 3)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, s))
    assert m["tasa_base"] == pytest.approx(0.3)


def test_roc_auc_intra_entidad_regla_da_medio_y_coincide_con_sklearn():
    b = _base_sintetica()
    tr, te = E.split_temporal(b, 2024)
    regla = E.regla_tasa_entidad(b.loc[tr, "nit_entidad"], b.loc[tr, "y"], b.loc[te, "nit_entidad"], 5)
    r = E.roc_auc_intra_entidad(b.loc[te, "y"], regla, b.loc[te, "nit_entidad"], n_boot=20)
    assert r["pares"] == pytest.approx(0.5) and r["por_entidad"] == pytest.approx(0.5)
    assert r["pares_ic"][0] == pytest.approx(0.5)
    # una sola entidad: igual al ROC-AUC de sklearn
    rng = np.random.default_rng(1)
    y, s = rng.integers(0, 2, 200), rng.random(200)
    assert E.roc_auc_intra_entidad(y, s, np.zeros(200))["pares"] == pytest.approx(roc_auc_score(y, s))


def test_elegir_m_usa_solo_train_y_devuelve_tabla():
    b = _base_sintetica()
    tr, _ = E.split_temporal(b, 2024)
    m, t = E.elegir_m(b.loc[tr, "nit_entidad"], b.loc[tr, "y"], b.loc[tr, "anio_firma"], tabla=True)
    assert m in E.M_CANDIDATOS and t["elegida"].sum() == 1
    assert t.loc[t["elegida"], "roc_auc"].iloc[0] >= t["roc_auc"].max() - E.TOLERANCIA_ROC_M


def test_comparar_referencia_dif_cero_y_modelo_perfecto_gana():
    b = _base_sintetica()
    tr, te = E.split_temporal(b, 2024)
    y = b.loc[te, "y"].to_numpy()
    ent = b.loc[te, "nit_entidad"].to_numpy()
    regla = E.regla_tasa_entidad(b.loc[tr, "nit_entidad"], b.loc[tr, "y"], ent, 5)
    perfecto = y + np.random.default_rng(0).random(len(y)) * 0.1
    t = E.comparar(y, {"regla_entidad": regla, "perfecto": perfecto}, ent, n_boot=30)
    ref = t[t["modelo"] == "regla_entidad"]
    assert np.allclose(ref["dif"].fillna(0), 0)
    assert E.cumple_criterio(t, "perfecto")
    assert not E.cumple_criterio(t, "regla_entidad")
    fila = t[(t["modelo"] == "perfecto") & (t["metrica"] == "pr_auc")].iloc[0]
    assert fila["ic_inf"] <= fila["valor"] <= fila["ic_sup"]


def test_evaluar_con_estimador_dummy():
    from sklearn.dummy import DummyClassifier
    b = _base_sintetica()
    X = pd.DataFrame({"x": np.zeros(len(b))}, index=b.index)
    r = E.evaluar(DummyClassifier(strategy="prior"), b, 2024, X=X, n_boot=5)
    assert set(r["scores"].columns) >= {"regla_base", "regla_entidad", "modelo", "y"}
    assert set(r["por_grupo"]["tipo"]) == {"entidad", "duracion"}
    assert r["m"] in E.M_CANDIDATOS
