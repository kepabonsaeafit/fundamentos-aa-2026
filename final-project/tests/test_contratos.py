"""Las tres verificaciones de los contratos entre las partes (plan, seccion 2d).

1. El esquema de la base.
2. FEATURES y las columnas de entrada no tocan la lista de fuga.
3. construir_features da lo mismo con entrada de texto (como la API) que con entrada tipada.

Corren sin datos: usan filas sinteticas con el formato de la API. Si existe data/processed/base_contratos.parquet,
tambien se revisa esa base. Ejecutar desde final-project/:  pytest
"""
import numpy as np
import pandas as pd
import pytest

from src import base as B
from src import config
from src import features as F
from src.uniones import elegir_proceso, unir_contrato_proceso

RUTA_BASE = config.DATA_PROCESSED / "base_contratos.parquet"


def _filas_api():
    """Contratos y Procesos como los devuelve la API: todo texto, fechas ISO, faltantes ausentes, con casos borde."""
    rng = np.random.default_rng(config.SEED)
    n = 14
    fechas = pd.to_datetime("2024-01-10") + pd.to_timedelta(rng.integers(0, 330, n), unit="D")
    iso = lambda f: f.strftime("%Y-%m-%dT00:00:00.000")
    contratos = pd.DataFrame({
        "id_contrato": [f"CO1.PCCNTR.{1000 + i}" for i in range(n)],
        "proceso_de_compra": [f"CO1.BDOS.{2000 + i}" for i in range(n)],
        "nombre_entidad": [f"Entidad {i % 4}" for i in range(n)],
        "dias_adicionados": rng.choice(["0", "30", "90"], n),
        "fecha_de_firma": fechas.map(iso),
        "fecha_de_inicio_del_contrato": (fechas + pd.to_timedelta(rng.integers(0, 20, n), unit="D")).map(iso),
        "codigo_de_categoria_principal": rng.choice(["V1.80111600", "V1.72101500", "UNSPECIFIED"], n),
        "objeto_del_contrato": [f"Prestar servicios profesionales numero {i}" for i in range(n)],
        "tipo_de_contrato": rng.choice(["Prestación de servicios", "Obra", "Suministros"], n),
        "modalidad_de_contratacion": rng.choice(["Contratación directa", "Licitación pública"], n),
        "justificacion_modalidad_de": rng.choice(["Servicios profesionales", "Mínima cuantía"], n),
        "origen_de_los_recursos": "Recursos propios",
        "destino_gasto": rng.choice(["Funcionamiento", "Inversión"], n),
        "espostconflicto": rng.choice(["Si", "No"], n),
        "es_grupo": "No", "es_pyme": rng.choice(["Si", "No"], n),
        "habilita_pago_adelantado": rng.choice(["Si", "No"], n),
        "valor_de_pago_adelantado": rng.choice(["0", "1500000", "25000000"], n),
        "documentos_tipo": "No", "el_contrato_puede_ser_prorrogado": rng.choice(["Si", "No"], n),
        "tipodocproveedor": rng.choice(["Cédula de Ciudadanía", "NIT"], n),
        "nit_entidad": rng.choice(["800214750", "890399010", "899999061"], n),
        "sector": "Salud", "orden": "Territorial", "rama": "Ejecutivo", "entidad_centralizada": "Centralizada",
        "departamento": "Antioquia", "ciudad": "Medellín",
    })
    p = pd.DataFrame({
        "id_del_portafolio": [f"CO1.BDOS.{2000 + i}" for i in range(n)],
        "id_del_proceso": [f"CO1.REQ.{3000 + i}" for i in range(n)],
        "adjudicado": "Si", "fecha_adjudicacion": fechas.map(iso), "valor_total_adjudicacion": "5000000",
        "precio_base": rng.choice(["5000000", "120000000", "3.5E7"], n),
        "duracion": rng.choice(["3", "6", "300"], n),
        "unidad_de_duracion": rng.choice(["Mes(es)", "Día(s)", "Semana(s)", "Año(s)"], n),
        "fecha_de_publicacion_del": (fechas - pd.to_timedelta(rng.integers(1, 40, n), unit="D")).map(iso),
        "numero_de_lotes": "1", "proveedores_invitados": rng.choice(["0", "3", "12"], n),
        "proveedores_con_invitacion": "0", "proveedores_que_manifestaron": rng.choice(["0", "2"], n),
        "respuestas_al_procedimiento": "1", "respuestas_externas": "0",
        "conteo_de_respuestas_a_ofertas": "0", "proveedores_unicos_con": "1",
    })
    # casos borde: sin proceso, proceso duplicado, anio mal digitado, campos faltantes, unidad que no se convierte
    contratos.loc[0, "proceso_de_compra"] = "CO1.BDOS.999999"
    contratos.loc[1, "codigo_de_categoria_principal"] = None
    contratos.loc[2, "fecha_de_inicio_del_contrato"] = "0025-05-01T00:00:00.000"
    contratos.loc[3, ["objeto_del_contrato", "valor_de_pago_adelantado"]] = [None, None]
    p.loc[4, "unidad_de_duracion"] = "Hora(s)"
    p.loc[5, ["precio_base", "duracion"]] = [None, None]
    duplicado = p.iloc[[6]].copy()
    duplicado["id_del_proceso"] = "CO1.REQ.4006"
    duplicado["adjudicado"] = "No"
    return contratos, pd.concat([p, duplicado], ignore_index=True)


def _a_texto_api(df):
    """Pasa una base tipada a texto como lo devuelve la API: enteros sin decimales, fechas ISO, faltantes ausentes."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[c] = s.dt.strftime("%Y-%m-%dT%H:%M:%S.000")
        elif pd.api.types.is_float_dtype(s):
            out[c] = s.map(lambda v: f"{v:.0f}" if v == int(v) else repr(v), na_action="ignore")
        else:
            out[c] = s.astype("object")
    return out.astype("object").where(out.notna(), np.nan)


# ---------------------------------------------------------------- 1. esquema de la base
def test_esquema_de_la_base():
    contratos, procesos = _filas_api()
    base = B.validar_esquema(B.construir_base(contratos, procesos, B.FECHA_CORTE_V0))
    assert len(base) == len(contratos) and base["id_contrato"].is_unique
    assert list(base.columns) == B.COLUMNAS_BASE
    assert not base.loc[base["id_contrato"] == "CO1.PCCNTR.1000", "eval_en_procesos"].any()   # el contrato sin proceso

    with pytest.raises(ValueError, match="fuga"):          # control negativo: una columna de fuga rompe el esquema
        B.validar_esquema(base.assign(valor_del_contrato=1.0))
    with pytest.raises(ValueError, match="unico"):
        B.validar_esquema(pd.concat([base, base.iloc[[0]]], ignore_index=True))

    if RUTA_BASE.exists():
        B.validar_esquema(pd.read_parquet(RUTA_BASE))


# ---------------------------------------------------------------- 2. sin fuga
def test_features_y_entradas_no_son_fuga():
    assert not set(F.FEATURES) & F.COLUMNAS_FUGA
    assert not set(F.FEATURES_EXCLUIDAS) & F.COLUMNAS_FUGA
    assert not set(F.ENTRADAS) & F.COLUMNAS_FUGA
    assert not (set(B.COLUMNAS_BASE) - set(B.ETIQUETAS)) & F.COLUMNAS_FUGA


# ---------------------------------------------------------------- 3. paridad texto (API) frente a tipado (entrenamiento)
def _comparar_texto_y_tipado(crudo_texto, crudo_tipado):
    x_texto = F.construir_features(crudo_texto)
    x_tipado = F.construir_features(crudo_tipado)
    assert list(x_texto.columns) == F.FEATURES
    assert dict(x_texto.dtypes) == dict(x_tipado.dtypes) == {c: F._TIPO[c] for c in F.FEATURES}
    pd.testing.assert_frame_equal(x_texto, x_tipado)
    return x_texto


def test_construir_features_igual_con_texto_y_con_tipos():
    contratos, procesos = _filas_api()
    unido_texto = unir_contrato_proceso(contratos, elegir_proceso(procesos))
    x = _comparar_texto_y_tipado(unido_texto, B.tipar_entradas(unido_texto))

    # una fila sola (como la app) da lo mismo que dentro del lote
    for i in [0, 2, 5, 7]:
        pd.testing.assert_frame_equal(F.construir_features(unido_texto.iloc[[i]]), x.iloc[[i]])

    if RUTA_BASE.exists():   # la base real, pasada a texto como la API
        muestra = pd.read_parquet(RUTA_BASE, columns=F.ENTRADAS).sample(300, random_state=config.SEED)
        _comparar_texto_y_tipado(_a_texto_api(muestra), muestra)
