"""Monitoreo y reajuste (G3), con datos sinteticos y sin leer la base: PSI, madurez, metricas, bootstrap por entidad,
reglas de alarma con persistencia, flujo trimestral y regla de promocion. Ejecutar desde final-project/:  pytest
"""
import numpy as np
import pandas as pd
import pytest

from src import monitoreo as M

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- madurez
def test_madura_en_replica_la_regla_de_la_base():
    firma = pd.Timestamp("2025-01-10")
    base = pd.DataFrame({
        "fecha_de_firma": [firma] * 4,
        "eval_dur_planeada_dias": [100, 100, 0, np.nan],
        "eval_fin_planeado": [firma + pd.Timedelta(days=100)] * 2 + [pd.NaT, pd.NaT],
    })
    # con duracion: fin + 60 dias; sin duracion (0 o faltante): firma + 365 dias
    assert M.madura_en(base, firma + pd.Timedelta(days=160)).tolist() == [True, True, False, False]
    assert M.madura_en(base, firma + pd.Timedelta(days=159)).tolist() == [False, False, False, False]
    assert M.madura_en(base, firma + pd.Timedelta(days=365)).tolist() == [True, True, True, True]


# ---------------------------------------------------------------- PSI
def test_psi_numerica_es_cero_sin_cambio_y_grande_con_cambio():
    ref = rng.normal(0, 1, 20_000)
    assert M.psi_numerica(ref, rng.normal(0, 1, 5_000)) < 0.02
    assert M.psi_numerica(ref, rng.normal(1.5, 1, 5_000)) > M.PSI_ALARMA


def test_psi_numerica_trata_los_faltantes_como_un_grupo():
    ref = np.concatenate([rng.normal(0, 1, 9_000), np.full(1_000, np.nan)])
    sin_faltantes = rng.normal(0, 1, 5_000)
    assert M.psi_numerica(ref, sin_faltantes) > M.PSI_AVISO       # de 10 % de faltantes a 0 %: es un cambio


def test_psi_categorica_ve_las_categorias_nuevas():
    ref = pd.Series(rng.choice(["a", "b", "c"], 10_000, p=[0.5, 0.3, 0.2]))
    igual = pd.Series(rng.choice(["a", "b", "c"], 5_000, p=[0.5, 0.3, 0.2]))
    nueva = pd.Series(rng.choice(["a", "b", "c", "z"], 5_000, p=[0.3, 0.2, 0.1, 0.4]))
    assert M.psi_categorica(ref, igual) < 0.02
    assert M.psi_categorica(ref, nueva) > M.PSI_ALARMA


def test_nivel_psi():
    assert [M.nivel_psi(v) for v in (0.05, 0.10, 0.20, 0.25, 0.9)] == ["normal", "aviso", "aviso", "alarma", "alarma"]


# ---------------------------------------------------------------- metricas
def test_metricas_con_un_puntaje_perfecto():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1] * 50)
    m = M.metricas(y, y.astype(float), entidad=np.arange(len(y)) % 10)
    assert m["roc_auc"] == 1 and m["pr_auc"] == 1 and m["lift"] == pytest.approx(1 / y.mean())
    assert m["precision_top"] == 1 and m["n"] == 500 and m["tasa_base"] == pytest.approx(0.2)
    assert 0 < m["concentracion_top"] <= 1


def test_metricas_sin_las_dos_clases_no_calcula_auc():
    m = M.metricas(np.zeros(100), rng.random(100))
    assert "roc_auc" not in m and m["precision_top"] == 0


# ---------------------------------------------------------------- bootstrap por entidad
def _periodo(n, ent, sep):
    """Contratos de `ent` entidades; `sep` mide cuanto separa el puntaje a los positivos."""
    entidad = rng.integers(0, ent, n)
    y = rng.random(n) < 0.3
    score = rng.normal(0, 1, n) + sep * y
    return pd.DataFrame({"y": y.astype(int), "score": score, "entidad": entidad})


def test_diferencia_entre_periodos_detecta_una_caida_real_y_no_inventa_una():
    a = _periodo(4_000, 80, sep=1.5)
    b_igual = _periodo(4_000, 80, sep=1.5)
    b_peor = _periodo(4_000, 80, sep=0.3)
    sin_cambio = M.diferencia_entre_periodos(a, b_igual, "roc_auc", n_boot=60)
    caida = M.diferencia_entre_periodos(a, b_peor, "roc_auc", n_boot=60)
    assert sin_cambio["ic_inf"] < 0 < sin_cambio["ic_sup"]            # el IC incluye 0
    assert caida["delta"] < -0.15 and caida["ic_sup"] < 0              # el IC excluye 0


def test_el_intervalo_remuestrea_entidades_y_por_eso_es_mas_ancho_que_remuestrear_contratos():
    # contratos de una misma entidad comparten su comportamiento: pocos grupos grandes -> intervalo ancho
    ent = rng.integers(0, 6, 3_000)
    efecto = rng.normal(0, 1, 6)[ent]
    y = (rng.random(3_000) < 0.3).astype(int)
    df = pd.DataFrame({"y": y, "score": rng.normal(0, 1, 3_000) + efecto + y, "entidad": ent})
    por_entidad = M.diferencia_entre_periodos(df, df, "roc_auc", n_boot=80)
    ancho_entidad = por_entidad["ic_sup"] - por_entidad["ic_inf"]
    idx = rng.integers(0, len(df), (80, len(df)))
    from sklearn.metrics import roc_auc_score
    por_contrato = [roc_auc_score(df["y"].to_numpy()[i], df["score"].to_numpy()[i]) for i in idx]
    assert ancho_entidad > 0 and np.ptp(np.quantile(por_contrato, [0.025, 0.975])) < 4 * ancho_entidad


def test_campeon_retador_mide_sobre_los_mismos_contratos():
    n = 5_000
    ent = rng.integers(0, 100, n)
    y = (rng.random(n) < 0.3).astype(int)
    campeon = rng.normal(0, 1, n) + 0.5 * y
    retador = rng.normal(0, 1, n) + 1.5 * y
    r = M.diferencia_campeon_retador(y, ent, campeon, retador, "roc_auc", n_boot=60)
    assert r["delta"] > 0.1 and r["ic_inf"] > 0
    igual = M.diferencia_campeon_retador(y, ent, campeon, campeon, "roc_auc", n_boot=20)
    assert igual["delta"] == 0 and igual["ic_inf"] == 0 == igual["ic_sup"]


def test_decision_de_promocion():
    def d(delta, ic_inf, ic_sup):
        return {"delta": delta, "ic_inf": ic_inf, "ic_sup": ic_sup}
    assert M.decision_promocion(d(0.03, 0.01, 0.05), d(0.0, -0.01, 0.01)) == "promover"
    assert M.decision_promocion(d(0.03, 0.01, 0.05), d(-0.05, -0.08, -0.03)).startswith("no promover")   # mejora el PR-AUC pero empeora claro el ROC
    assert M.decision_promocion(d(0.01, -0.02, 0.04), d(0.0, -0.01, 0.01)).startswith("no promover")      # IC incluye 0
    assert M.decision_promocion(d(-0.04, -0.06, -0.02), d(0.0, -0.01, 0.01)).startswith("descartar")


# ---------------------------------------------------------------- persistencia
def test_nivel_con_persistencia():
    avisos = [False, True, True, True, False, True]
    assert M.nivel_con_persistencia(avisos) == ["normal", "aviso", "alarma", "alarma", "normal", "aviso"]
    assert M.nivel_con_persistencia([True, True, True], persistencia=3) == ["aviso", "aviso", "alarma"]


# ---------------------------------------------------------------- flujo trimestral
# Con 2.000 contratos por trimestre el error estandar de una tasa de 0,30 es ~1 punto: el umbral de 5 puntos no se cruza por azar.
N_POR_TRIM = 2_000


def _base_sintetica(corte):
    """Contratos de 2023 a 2025 (N_POR_TRIM por trimestre) con duracion de 30 dias; el puntaje separa los positivos."""
    filas = []
    for q in pd.period_range("2023Q1", "2025Q4", freq="Q"):
        ini = q.start_time
        fecha = ini + pd.to_timedelta(rng.integers(0, 85, N_POR_TRIM), unit="D")
        y_dias_rate = 0.12 if q.year < 2025 else 0.25          # salto de registro en 2025
        y = (rng.random(N_POR_TRIM) < 0.3).astype(int)
        filas.append(pd.DataFrame({
            "fecha_de_firma": fecha, "anio_firma": q.year, "nit_entidad": rng.integers(0, 40, N_POR_TRIM).astype(str),
            "y": y, "y_dias": (rng.random(N_POR_TRIM) < y_dias_rate).astype(int),
            "eval_dur_planeada_dias": 30.0, "eval_fin_planeado": fecha + pd.Timedelta(days=30),
        }))
    base = pd.concat(filas, ignore_index=True)
    base["score"] = rng.normal(0, 1, len(base)) + 1.2 * base["y"]
    return base


def test_trimestral_evalua_solo_cohortes_completas_y_marca_el_cambio_de_registro():
    corte = pd.Timestamp("2025-12-31")
    base = _base_sintetica(corte)
    t = M.tabla_trimestral(base, base["score"].to_numpy(), corte=corte, n_boot=30).set_index("trimestre")
    assert list(t.index) == [str(q) for q in pd.period_range("2024Q1", "2025Q4", freq="Q")]
    assert t.loc["2025Q4", "madurez"] < M.MADUREZ_MIN and not t.loc["2025Q4", "completo"]
    assert np.isnan(t.loc["2025Q4", "roc_auc"])                       # trimestre incompleto: sin etiquetas
    assert t.loc["2024Q2", "completo"] and t.loc["2024Q2", "roc_auc"] > 0.6
    assert np.isnan(t.loc["2024Q2", "delta_roc_auc"])                 # 2024 no se compara en desempeno con 2023 (entrenamiento)
    assert not np.isnan(t.loc["2025Q2", "delta_roc_auc"])             # 2025 si, contra el mismo trimestre de 2024
    # y_dias salta en 2025 y la etiqueta y no: solo la primera avisa
    assert t.loc["2025Q1":"2025Q3", "aviso_tasa_y_dias"].all()
    assert not t.loc["2025Q1":"2025Q3", "aviso_tasa_y"].any()
    assert (t["estado_desempeno"] == "normal").all()                  # el puntaje no cambio: sin alarma de desempeno


def test_el_top_peor_que_el_azar_se_marca():
    corte = pd.Timestamp("2025-12-31")
    base = _base_sintetica(corte)
    q3 = (base["fecha_de_firma"].dt.to_period("Q") == "2025Q3").to_numpy()
    # en 2025Q3 los puntajes mas altos pasan a ser negativos reales: la lista de mayor riesgo falla
    base.loc[q3, "score"] = np.where(base.loc[q3, "y"] == 1, rng.normal(-1, 0.1, q3.sum()), rng.normal(3, 0.1, q3.sum()))
    t = M.tabla_trimestral(base, base["score"].to_numpy(), corte=corte, n_boot=20).set_index("trimestre")
    assert t.loc["2025Q3", "aviso_top_peor_que_base"] and not t.loc["2025Q1", "aviso_top_peor_que_base"]
