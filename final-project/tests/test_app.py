"""Consulta de un contrato y explicacion del puntaje (G1), sin red: cliente falso y datos sinteticos.

Cubre lo que no puede fallar en silencio: que no se pidan columnas de fuga ni de datos personales, que el id pegado se
interprete bien y no se pueda inyectar en la consulta, que las fechas con hora no corran un dia las features, que la
app caiga al respaldo si la API falla, y que SHAP sea aditivo con el modelo dummy. Ejecutar desde final-project/:  pytest
"""
import numpy as np
import pandas as pd
import pytest
import requests
from sklearn.ensemble import HistGradientBoostingClassifier

from src import api_secop as A
from src import config, explicacion as E, features as F, modelo_io
from src.cliente_socrata import Throttled

ID = "CO1.PCCNTR.1234567"


def _contrato(**cambios):
    """Una fila de Contratos como la devuelve la API: todo en texto y las fechas con hora."""
    fila = {c: None for c in A.COLS_CONTRATO_API}
    fila.update({
        "id_contrato": ID, "proceso_de_compra": "CO1.BDOS.7654321", "nombre_entidad": "MUNICIPIO DE PRUEBA",
        "fecha_de_firma": "2024-03-05T14:30:00.000", "fecha_de_inicio_del_contrato": "2024-03-06T00:00:00.000",
        "codigo_de_categoria_principal": "V1.72101500", "objeto_del_contrato": "Mantenimiento de vias urbanas",
        "tipo_de_contrato": "Obra", "modalidad_de_contratacion": "Licitación pública", "nit_entidad": "890000001",
        "departamento": "Antioquia", "ciudad": "Medellín", "sector": "Territorial", "orden": "Territorial",
        "rama": "Ejecutivo", "entidad_centralizada": "Centralizada", "origen_de_los_recursos": "Recursos Propios",
        "habilita_pago_adelantado": "No", "documentos_tipo": "No", "es_pyme": "No", "es_grupo": "No",
    })
    fila.update(cambios)
    return pd.DataFrame([fila])


def _procesos():
    filas = []
    for lote in (1, 2):    # un proceso con dos filas (lotes): elegir_proceso deja una
        fila = {c: None for c in A.COLS_PROCESO_API}
        fila.update({"id_del_portafolio": "CO1.BDOS.7654321", "id_del_proceso": f"CO1.REQ.{lote}", "adjudicado": "Si",
                     "precio_base": "500000000", "duracion": "6", "unidad_de_duracion": "Meses",
                     "fecha_de_publicacion_del": "2024-02-01T00:00:00.000", "numero_de_lotes": "2"})
        filas.append(fila)
    return pd.DataFrame(filas)


class ClienteFalso:
    def __init__(self, contratos=None, procesos=None, error=None):
        self.respuestas = {config.DATASETS["contratos"]: contratos, config.DATASETS["procesos"]: procesos}
        self.error = error
        self.llamadas = []

    def consultar(self, dataset_id, params):
        self.llamadas.append((dataset_id, params))
        if self.error:
            raise self.error
        datos = self.respuestas[dataset_id]
        return [] if datos is None else datos.to_dict("records")


@pytest.fixture(autouse=True)
def cache_limpia():
    A._CACHE.clear()


# ---------------------------------------------------------------- id y consulta segura
def test_extrae_el_id_de_texto_suelto_y_de_un_enlace():
    assert A.extraer_id_contrato("  co1.pccntr.1234567 ") == ID
    assert A.extraer_id_contrato(f"https://community.secop.gov.co/Public/x?doc={ID}&v=1") == ID
    for malo in ("", "hola", "CO1.BDOS.123", None):
        with pytest.raises(A.IdInvalido):
            A.extraer_id_contrato(malo)


def test_el_where_no_deja_pasar_comillas():
    assert A._donde_igual("id_contrato", ID) == f"id_contrato = '{ID}'"
    with pytest.raises(ValueError):
        A._donde_igual("id_contrato", "x' OR '1'='1")


def test_no_se_piden_columnas_de_fuga_ni_de_datos_personales():
    pedidas = set(A.COLS_CONTRATO_API) | set(A.COLS_PROCESO_API)
    assert not pedidas & F.COLUMNAS_FUGA
    assert not pedidas & config.PII
    cliente = ClienteFalso(_contrato(), _procesos())
    A.obtener_consulta(ID, cliente=cliente)
    for _, params in cliente.llamadas:
        assert not set(params["$select"].split(",")) & (F.COLUMNAS_FUGA | config.PII)


def test_token_de_secretos_nunca_falla():
    assert A.token_de({"SOCRATA_APP_TOKEN": "abc"}) == "abc"
    assert A.token_de({}) is None and A.token_de(None) is None and A.token_de({"SOCRATA_APP_TOKEN": ""}) is None

    class SinArchivo:      # como st.secrets cuando no existe secrets.toml
        def get(self, *_):
            raise FileNotFoundError("No secrets found")
    assert A.token_de(SinArchivo()) is None


def test_sin_token_no_se_manda_el_header_vacio(monkeypatch):
    enviados = {}
    monkeypatch.setattr(requests.Session, "request",
                        lambda self, method, url, **kw: enviados.update(kw) or "respuesta")
    sesion = A._SesionSinCabecerasVacias()
    sesion.get("https://www.datos.gov.co/resource/jbjy-vk9h.json", headers={"X-App-Token": "", "Accept": "application/json"})
    assert enviados["headers"] == {"Accept": "application/json"}
    sesion.get("https://www.datos.gov.co/resource/jbjy-vk9h.json", headers={"X-App-Token": "t", "Accept": "a"})
    assert enviados["headers"] == {"X-App-Token": "t", "Accept": "a"}


# ---------------------------------------------------------------- features desde la API
def test_las_features_salen_completas_y_con_los_mismos_dtypes_que_el_entrenamiento():
    c = A.consulta_desde_filas(_contrato(), _procesos())
    assert list(c.features.columns) == F.FEATURES
    assert len(c.features) == 1
    assert {col: str(tipo) for col, tipo in c.features.dtypes.items()} == {col: F._TIPO[col] for col in F.FEATURES}
    assert c.contrato["nombre_entidad"] == "MUNICIPIO DE PRUEBA"
    assert c.avisos == []


def test_la_hora_en_las_fechas_no_corre_un_dia_las_features():
    # firma a las 14:30 e inicio a las 00:00 del dia siguiente: son 1 dia, no 0 (truncando al dia, como en la base)
    c = A.consulta_desde_filas(_contrato(), _procesos())
    assert c.features.iloc[0]["dias_firma_a_inicio"] == 1
    assert c.features.iloc[0]["mes_firma"] == 3


def test_avisos_sin_proceso_y_fuera_de_periodo():
    sin_proceso = A.consulta_desde_filas(_contrato(), pd.DataFrame())
    assert A.AVISO_SIN_PROCESO in sin_proceso.avisos
    antiguo = A.consulta_desde_filas(_contrato(fecha_de_firma="2017-05-05T00:00:00.000"), _procesos())
    assert A.AVISO_FUERA_DE_PERIODO in antiguo.avisos


# ---------------------------------------------------------------- API caida y respaldo
def test_contrato_inexistente():
    with pytest.raises(A.ContratoNoEncontrado):
        A.obtener_consulta(ID, cliente=ClienteFalso(contratos=None))


def test_si_la_api_falla_y_no_hay_respaldo_el_error_es_claro():
    with pytest.raises(A.ApiNoDisponible):
        A.obtener_consulta(ID, cliente=ClienteFalso(error=Throttled("429")))


def test_si_la_api_falla_responde_desde_el_respaldo():
    c = A.consulta_desde_filas(_contrato(), _procesos())
    unido = _contrato().assign(**{k: v for k, v in _procesos().iloc[0].add_prefix("p_").items()})
    respaldo = unido.assign(eval_en_procesos=True)
    r = A.obtener_consulta(ID, cliente=ClienteFalso(error=RuntimeError("sin red")), respaldo=respaldo)
    assert r.fuente == "respaldo"
    pd.testing.assert_frame_equal(r.features.reset_index(drop=True), c.features.reset_index(drop=True))
    # un contrato que no esta en el respaldo sigue fallando con el mensaje claro
    with pytest.raises(A.ApiNoDisponible):
        A.obtener_consulta("CO1.PCCNTR.999", cliente=ClienteFalso(error=RuntimeError("sin red")), respaldo=respaldo)


def test_el_respaldo_por_ruta_lee_solo_la_fila_pedida(tmp_path):
    unido = _contrato().assign(**{k: v for k, v in _procesos().iloc[0].add_prefix("p_").items()})
    otro = unido.assign(id_contrato="CO1.PCCNTR.7654321")
    ruta = tmp_path / "respaldo.parquet"
    pd.concat([unido, otro]).assign(eval_en_procesos=True).to_parquet(ruta)
    c = A.obtener_consulta(ID, cliente=ClienteFalso(error=RuntimeError("sin red")), respaldo=ruta)
    assert c.fuente == "respaldo" and c.id_contrato == ID and len(c.features) == 1
    with pytest.raises(A.ApiNoDisponible):      # id que no esta en el archivo
        A.obtener_consulta("CO1.PCCNTR.999", cliente=ClienteFalso(error=RuntimeError("sin red")), respaldo=ruta)
    with pytest.raises(A.ApiNoDisponible):      # la ruta no existe
        A.obtener_consulta(ID, cliente=ClienteFalso(error=RuntimeError("sin red")), respaldo=tmp_path / "no_existe.parquet")


def test_la_consulta_repetida_sale_del_cache():
    cliente = ClienteFalso(_contrato(), _procesos())
    A.obtener_consulta(ID, cliente=cliente)
    n = len(cliente.llamadas)
    A.obtener_consulta(ID, cliente=cliente)
    assert len(cliente.llamadas) == n


# ---------------------------------------------------------------- explicacion
@pytest.fixture(scope="module")
def dummy():
    pipeline, meta = modelo_io.cargar_modelo(config.MODELS / "modelo_prorroga_dummy.joblib")
    return pipeline, meta, E.Explicador(pipeline)


def test_shap_es_aditivo_con_el_modelo_dummy(dummy):
    pipeline, _, explicador = dummy
    c = A.consulta_desde_filas(_contrato(), _procesos())
    e = explicador.explicar(c.features, c.contrato)
    assert e.error_aditividad < E.TOLERANCIA_ADITIVIDAD
    assert set(e.aportes["feature"]) == set(F.FEATURES)
    assert e.aportes["aporte"].abs().is_monotonic_decreasing
    assert e.puntaje == pytest.approx(pipeline.predict_proba(c.features)[0, 1])


def test_el_explicador_se_niega_con_categorias_nativas():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.integers(0, 3, 200), "b": rng.normal(size=200)})
    y = (X["a"] == 1).astype(int)
    modelo = HistGradientBoostingClassifier(categorical_features=[0], random_state=0).fit(X, y)
    from sklearn.pipeline import Pipeline
    with pytest.raises(E.ModeloNoSoportado):
        E.Explicador(Pipeline([("modelo", modelo)]), features=["a", "b"])


def test_nivel_y_percentil(dummy):
    _, meta, _ = dummy
    medio, alto = meta["umbrales_riesgo"]["medio"], meta["umbrales_riesgo"]["alto"]
    assert E.nivel_y_percentil(medio / 2, meta)[0] == "bajo"
    assert E.nivel_y_percentil((medio + alto) / 2, meta)[0] == "medio"
    assert E.nivel_y_percentil(alto * 2, meta)[0] == "alto"
    p_bajo, p_alto = (E.nivel_y_percentil(x, meta)[1] for x in (0.0, 1.0))
    assert 0 <= p_bajo < p_alto <= 100


def test_valores_en_lenguaje_claro():
    assert E.valor_legible("mes_firma", 3) == "marzo"
    assert E.valor_legible("dia_semana", 0) == "lunes"
    assert E.valor_legible("dias_firma_a_inicio", 1) == "1 día"
    assert E.valor_legible("p_dur_dias", 300) == "300 días"
    assert E.valor_legible("log_precio_base", np.log1p(960_400_000)) == "$ 960,4 millones"
    assert E.valor_legible("fin_planeado_cruza_anio", 1.0) == "Sí"
    assert E.valor_legible("nit_entidad", "890000001", {"nombre_entidad": "MUNICIPIO DE PRUEBA"}) == "MUNICIPIO DE PRUEBA"
    assert E.valor_legible("dias_firma_a_inicio", np.nan) == E.SIN_DATO
