"""Pruebas del modelo v1 (src/modelo.py) y de su artefacto, con lo que piden la app y el monitoreo (G2-G3)."""
import json

import numpy as np
import pytest

from src import features as F
from src import modelo as MO
from src import modelo_io
from tests.test_preprocesamiento import _X


def test_construir_pipeline_sin_nativas_y_reproducible():
    X, y = _X()
    a = MO.construir_pipeline().fit(X, y)
    b = MO.construir_pipeline().fit(X, y)
    assert not np.any(a[-1].is_categorical_)
    assert a[-1].early_stopping is False
    assert np.allclose(a.predict_proba(X)[:, 1], b.predict_proba(X)[:, 1])
    assert hasattr(a[-1], "decision_function")       # la app verifica la aditividad en log-odds


@pytest.mark.skipif(not MO.RUTA_V1.exists(), reason="artefacto v1 no generado (python -m src.modelo)")
def test_artefacto_v1_cumple_el_contrato_de_la_app():
    pipe, meta = modelo_io.cargar_modelo(MO.RUTA_V1)
    assert meta["version"].startswith("v1")
    assert {"medio", "alto"} <= set(meta["umbrales_riesgo"]) and meta["umbrales_riesgo"]["medio"] < meta["umbrales_riesgo"]["alto"]
    p = meta["percentiles_puntaje"]
    assert len(p) == 101 and p == sorted(p)
    assert "2024" in meta["percentiles_fuente"]
    assert len(meta["entidades_vistas"]) > 0 and all(isinstance(e, str) for e in meta["entidades_vistas"])
    assert meta["features"] == F.FEATURES and meta["modelo"]["categorias_nativas"] is False
    assert not np.any(pipe[-1].is_categorical_)
    json.dumps(meta)
