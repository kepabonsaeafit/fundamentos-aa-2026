"""Chequeo de id_contrato repetidos en Descarga.chequear_contratos, lotes con ids unicos y reutilizacion de unidades.
Sin red: el cliente falso falla ante cualquier peticion y el count(*) del servidor se sustituye por lo descargado por
mes. Ejecutar desde final-project/:  pytest tests/test_descarga_duplicados.py
"""
import argparse
import json

import pandas as pd
import pytest

from src import descarga as D


class ClienteFalso:
    base_url = "https://datos.gov.co"

    def __init__(self, *args, **kwargs):
        self.peticiones, self.registro = 0, []

    def consultar(self, *args, **kwargs):
        raise AssertionError("no debe haber peticiones a la red")


@pytest.fixture
def desc(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "leer_token", lambda: "token-de-prueba")
    monkeypatch.setattr(D, "ClienteSocrata", ClienteFalso)
    d = D.Descarga(argparse.Namespace(digito="4", mes=None, destino=str(tmp_path), corte=None, hora_limite=None))
    d.desde, d.hasta = "2019-01", "2019-02"   # dos unidades bastan
    d.conteo_servidor = lambda unidades: {u["mes"]: len(pd.read_parquet(u["ruta"])) for u in unidades}
    return d


def fila(id_, u, **cambios):
    r = {c: "x" for c in D.COLS["contratos"]}
    r.update(id_contrato=id_, fecha_de_firma=f"{u['ini']:%Y-%m-%d}T00:00:00.000")
    r.update(cambios)
    return r


def escribir(unidades, filas_por_unidad):
    for u, filas in zip(unidades, filas_por_unidad):
        D.guardar_parquet(filas, D.COLS["contratos"], u["ruta"])


def dos_unidades(desc, en_a, en_b=()):
    """Escribe las dos unidades con las filas dadas (funciones u -> fila) y las devuelve."""
    unidades = desc.unidades_contratos()
    escribir(unidades, [[f(unidades[0]) for f in en_a], [f(unidades[1]) for f in en_b]])
    return unidades


def fila_de(id_, **cambios):
    return lambda u: fila(id_, u, **cambios)


def resultado(desc, unidades):
    ok = desc.chequear_contratos(unidades)
    r = desc.estado.datos["chequeos"]["contratos"]
    assert r["ok"] == ok
    json.dumps(r)   # completitud.json lo escribe tal cual
    assert r["duplicados_id_contrato"] == (r["duplicados_exactos_id_contrato"] + r["duplicados_no_exactos"]
                                           + r["duplicados_entre_unidades"])
    return ok, r


def test_sin_duplicados_pasa(desc):
    unidades = dos_unidades(desc, [fila_de("A4"), fila_de("B4")], [fila_de("C4")])
    ok, r = resultado(desc, unidades)
    assert ok
    assert (r["duplicados_id_contrato"], r["duplicados_exactos_id_contrato"], r["duplicados_no_exactos"],
            r["duplicados_entre_unidades"], r["duplicados_exactos_detalle"]) == (0, 0, 0, 0, [])


def test_copias_exactas_dentro_de_la_unidad_pasan_y_se_registran(desc):
    # A4 dos veces y B4 tres veces en la unidad 1 (con una columna vacia, igual en todas las copias)
    unidades = dos_unidades(desc, [fila_de("A4", ciudad=None), fila_de("A4", ciudad=None), fila_de("B4"),
                                   fila_de("B4"), fila_de("B4")], [fila_de("C4")])
    ok, r = resultado(desc, unidades)
    assert ok
    assert (r["duplicados_id_contrato"], r["duplicados_exactos_id_contrato"], r["duplicados_no_exactos"],
            r["duplicados_entre_unidades"]) == (3, 3, 0, 0)
    assert r["duplicados_exactos_detalle"] == [
        {"id_contrato": "A4", "unidad": unidades[0]["clave"], "filas": 2, "copias_exactas_sobrantes": 1,
         "estado": "aceptados: copias exactas dentro de la unidad"},
        {"id_contrato": "B4", "unidad": unidades[0]["clave"], "filas": 3, "copias_exactas_sobrantes": 2,
         "estado": "aceptados: copias exactas dentro de la unidad"}]


def test_registra_en_el_log(desc, caplog):
    unidades = dos_unidades(desc, [fila_de("A4"), fila_de("A4")], [fila_de("C4")])
    with caplog.at_level("INFO", logger="descarga"):
        assert desc.chequear_contratos(unidades)
    texto = caplog.text
    assert "aceptados: copias exactas dentro de la unidad" in texto
    assert f"A4 en {unidades[0]['clave']}" in texto


def test_mismo_id_con_una_columna_distinta_falla(desc):
    unidades = dos_unidades(desc, [fila_de("A4", valor_del_contrato="1"), fila_de("A4", valor_del_contrato="2"),
                                   fila_de("E4"), fila_de("E4")], [fila_de("C4")])   # E4: copia exacta, no falla sola
    ok, r = resultado(desc, unidades)
    assert not ok
    assert (r["duplicados_id_contrato"], r["duplicados_exactos_id_contrato"], r["duplicados_no_exactos"],
            r["duplicados_entre_unidades"]) == (2, 1, 1, 0)
    assert [d["id_contrato"] for d in r["duplicados_exactos_detalle"]] == ["E4"]


def test_columna_vacia_frente_a_valor_no_es_copia_exacta(desc):
    unidades = dos_unidades(desc, [fila_de("A4", ciudad=None), fila_de("A4", ciudad="Cali")], [fila_de("C4")])
    ok, r = resultado(desc, unidades)
    assert not ok and r["duplicados_no_exactos"] == 1


def test_id_con_copias_exactas_y_una_fila_distinta_falla_entero(desc):
    unidades = dos_unidades(desc, [fila_de("A4"), fila_de("A4"), fila_de("A4", valor_del_contrato="2")],
                            [fila_de("C4")])
    ok, r = resultado(desc, unidades)
    assert not ok
    assert (r["duplicados_id_contrato"], r["duplicados_exactos_id_contrato"], r["duplicados_no_exactos"]) == (2, 0, 2)
    assert r["duplicados_exactos_detalle"] == []   # un id que falla no se marca como aceptado


@pytest.mark.parametrize("cambios", [{}, {"valor_del_contrato": "2"}])
def test_id_en_dos_unidades_falla(desc, cambios):
    unidades = dos_unidades(desc, [fila_de("A4"), fila_de("B4")], [fila_de("A4", **cambios)])
    ok, r = resultado(desc, unidades)
    assert not ok
    assert (r["duplicados_id_contrato"], r["duplicados_exactos_id_contrato"], r["duplicados_no_exactos"],
            r["duplicados_entre_unidades"]) == (1, 0, 0, 1)


def ids_de(lotes):
    return [i for l in lotes for i in l["ids"]]


def test_lotes_con_ids_unicos_y_mismo_sha1_con_copias_exactas(desc):
    n = 1200
    ids = [f"ID{k:05d}4" for k in range(n)]

    def filas(unidad, indices):   # dos contratos por proceso
        return [fila(ids[k], unidad, proceso_de_compra=f"P{k // 2}") for k in indices]

    unidades = desc.unidades_contratos()
    # 10 ids con una copia exacta dentro de su unidad: 0..6 en la primera, 600..602 en la segunda
    escribir(unidades, [filas(unidades[0], [*range(0, 600), *range(7)]),
                        filas(unidades[1], [*range(600, n), *range(600, 603)])])
    con_copias = {ds: desc.lotes(ds) for ds in ("modificaciones", "procesos")}
    escribir(unidades, [filas(unidades[0], range(0, 600)), filas(unidades[1], range(600, n))])
    sin_copias = {ds: desc.lotes(ds) for ds in ("modificaciones", "procesos")}
    for ds in ("modificaciones", "procesos"):
        todos = ids_de(con_copias[ds])
        assert len(todos) == len(set(todos)) and todos == sorted(todos)   # ningun id repetido, ni dentro ni entre lotes
        assert [l["clave"] for l in con_copias[ds]] == [l["clave"] for l in sin_copias[ds]]   # mismo nombre (sha1)
        assert [l["ids"] for l in con_copias[ds]] == [l["ids"] for l in sin_copias[ds]]
    assert len(ids_de(con_copias["modificaciones"])) == n and len(con_copias["modificaciones"]) == 3
    assert len(ids_de(con_copias["procesos"])) == n // 2


def test_unidades_hechas_se_reutilizan_sin_peticiones(desc):
    unidades = dos_unidades(desc, [fila_de("A4"), fila_de("A4")], [fila_de("C4")])
    assert desc.estado.planificar("contratos", unidades) == []   # los parquet existen: se marcan hechos, no se piden
    desc.recorrer("contratos", unidades, D.LIMITE_CONTRATOS, lambda u: u["where"])
    assert desc.bajadas["contratos"] == 0 and desc.peticiones() == 0
    assert [f["estado"] for f in desc.estado.filas("contratos")] == [D.HECHO, D.HECHO]
