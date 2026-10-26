"""Modo de Contratos local de src/descarga.py (base B), sin red: servidor falso y archivos reales de la exploracion.

Cubre: lectura de los 106 archivos y conteo de 167.997 contratos (los de la base v0), formato igual al de F2 normal, plan
de lotes, conversion del corte (B1: fecha <= 2026-10-14, sin pasar a UTC), corte y reanudacion de Procesos y
Modificaciones con el conteo del servidor por lote, y que el modo por defecto (sin --contratos-local) no cambia.
Ninguna prueba toca datos.gov.co: `requests.Session.get` lanza un error salvo que se instale el servidor falso.
Ejecutar desde final-project/:  pytest
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd
import pytest
import requests

from src import config, descarga
from src.cliente_socrata import LIMITADOR

TOKEN = "token-de-prueba-123"
LOCAL = descarga.CONTRATOS_LOCAL
hay_local = LOCAL.is_dir() and len(list(LOCAL.glob("*.parquet"))) == 106
requiere_local = pytest.mark.skipif(not hay_local, reason="faltan los 106 archivos de la exploracion")


class Respuesta:
    status_code = 200
    headers = {}

    def __init__(self, cuerpo):
        self._cuerpo, self.text = cuerpo, "ok"

    def json(self):
        return self._cuerpo


def filas_servidor(dataset, ids):
    """Filas que el servidor falso devuelve para una lista de IDs (deterministas, a partir del sha1 de cada ID)."""
    out = []
    for i in ids:
        h = int(hashlib.sha1(i.encode()).hexdigest()[:4], 16)
        if dataset == "procesos":
            out += [{"id_del_portafolio": i, "id_del_proceso": f"{i}.{k}", "adjudicado": "Si"} for k in range(1 + h % 2)]
        elif h % 3 == 0:
            out.append({"id_contrato": i, "identificador_modificacion": "1", "version_anterior_del_contrato": "0",
                        "fecha_version": "2026-10-14T00:00:00.000", "estado_modificacion": "Publicado",
                        "dias_extendidos": "30", "fecha_de_aprobacion": "2026-10-14T00:00:00.000",
                        "fecha_fin_contrato": "2027-01-01T00:00:00.000"})
    return out


class Servidor:
    """Imita /resource/<id>.json de datos.gov.co para los tres conjuntos y registra cada peticion."""

    def __init__(self, contratos=()):
        self.contratos = list(contratos)      # filas de Contratos (solo el modo por defecto las pide)
        self.registro = []                    # dicts: dataset, tipo (datos|conteo|conteo_mes), clave, where, params
        self.datos = 0
        self.interrumpir_en = None            # numero de peticion de datos que lanza KeyboardInterrupt
        self.sumar_al_conteo = 0              # se suma al count(*) de TODOS los lotes (prueba de desacuerdo)

    def get(self, url, params):
        dataset = next(k for k, v in config.DATASETS.items() if f"/{v}.json" in url)
        where, select = params["$where"], params.get("$select", "")
        if dataset == "contratos":
            if "date_trunc_ym" in select:
                self.registro.append({"dataset": dataset, "tipo": "conteo_mes", "params": dict(params)})
                return [{"mes": "2024-08-01T00:00:00.000", "n": str(len(self.contratos))}]
            self.registro.append({"dataset": dataset, "tipo": "datos", "params": dict(params)})
            return self.contratos[params["$offset"]:params["$offset"] + params["$limit"]]
        ids = re.findall(r"'([^']*)'", where.split(" AND ")[0])
        clave = "lote_" + descarga.sha1_ids(ids)[:20]
        filas = filas_servidor(dataset, ids)
        if select == "count(*) as n":
            self.registro.append({"dataset": dataset, "tipo": "conteo", "clave": clave, "where": where})
            return [{"n": str(len(filas) + self.sumar_al_conteo)}]
        self.datos += 1
        if self.interrumpir_en is not None and self.datos == self.interrumpir_en:
            raise KeyboardInterrupt
        self.registro.append({"dataset": dataset, "tipo": "datos", "clave": clave, "where": where,
                              "params": dict(params)})
        return filas[params["$offset"]:params["$offset"] + params["$limit"]]

    def claves(self, dataset, tipo):
        return [r["clave"] for r in self.registro if r["dataset"] == dataset and r["tipo"] == tipo]


@pytest.fixture(autouse=True)
def entorno(monkeypatch):
    monkeypatch.setenv("SOCRATA_APP_TOKEN", TOKEN)
    monkeypatch.setattr(LIMITADOR, "intervalo", 0.0)
    monkeypatch.setattr(descarga.time, "sleep", lambda s: None)

    def sin_red(self, *a, **k):
        raise AssertionError("peticion de red en una prueba sin red")
    monkeypatch.setattr(requests.Session, "get", sin_red)
    yield
    for h in list(descarga.log.handlers):
        h.close()
    descarga.log.handlers.clear()


def instalar(monkeypatch, servidor):
    monkeypatch.setattr(requests.Session, "get",
                        lambda self, url, params=None, headers=None, timeout=None: Respuesta(servidor.get(url, params)))


def argumentos(destino, **kw):
    base = dict(mes=None, digito=None, destino=str(destino), corte=None, hora_limite=None, contratos_local=None)
    base.update(kw)
    return argparse.Namespace(**base)


def lanzar(argv):
    """Corre main() y devuelve (codigo, completitud.json)."""
    codigo = descarga.main(argv)
    destino = argv[argv.index("--destino") + 1]
    return codigo, json.loads((Path(destino) / "completitud.json").read_text(encoding="utf-8"))


# ---------------- carpeta local pequena (rapida) ----------------
def fila_contrato(i, mes, proceso=None):
    fila = {c: f"valor_{c}_{i}" for c in descarga.COLS["contratos"]}
    fila.update(id_contrato=f"CO1.PCCNTR.{i}", proceso_de_compra=proceso or f"CO1.BDOS.{i}",
                fecha_de_firma=f"{mes}-15T00:00:00.000", codigo_proveedor="NO DEBE PASAR", ultima_actualizacion="x")
    fila["objeto_del_contrato"] = None
    return fila


@pytest.fixture
def carpeta_pequena(tmp_path, monkeypatch):
    """2018-12 y 2026-01 (fuera de rango), 2024-07 (una fila repetida) y 2024-08; rango de prueba 2024-07 a 2024-08."""
    carpeta = tmp_path / "chunks"
    carpeta.mkdir()
    contenido = {"2018-12": [fila_contrato(1, "2018-12")], "2026-01": [fila_contrato(2, "2026-01")],
                 "2024-07": [fila_contrato(i, "2024-07", f"CO1.BDOS.{i // 2}") for i in range(10, 16)],
                 "2024-08": [fila_contrato(i, "2024-08") for i in range(20, 26)]}
    contenido["2024-07"].append(dict(contenido["2024-07"][0]))     # repetida por el servidor
    for mes, filas in contenido.items():
        pd.DataFrame(filas).astype("string").to_parquet(carpeta / f"{mes}.parquet", index=False)
    monkeypatch.setattr(descarga, "DESDE", "2024-07")
    monkeypatch.setattr(descarga, "HASTA", "2024-08")
    monkeypatch.setattr(descarga, "CONTRATOS_LOCAL_ESPERADOS", 12)
    return carpeta


def test_carpeta_pequena_columnas_repetidas_y_descartes(carpeta_pequena, tmp_path):
    d = descarga.Descarga(argumentos(tmp_path / "b", contratos_local=str(carpeta_pequena), corte=descarga.CORTE_LOCAL))
    unidades = d.unidades_contratos()
    assert [u["clave"] for u in unidades] == ["2024-07_local", "2024-08_local"]
    d.copiar_contratos_local(unidades)
    assert d.chequear_contratos_local(unidades) is True
    ruta = tmp_path / "b" / "contratos" / "anio=2024" / "mes=07_local.parquet"
    df = pd.read_parquet(ruta)
    assert list(df.columns) == descarga.COLS["contratos"]               # sin codigo_proveedor ni ultima_actualizacion
    assert len(df) == 6 and df["id_contrato"].is_unique                  # la repetida se elimino
    assert df["objeto_del_contrato"].isna().all()                        # los nulos siguen siendo nulos
    plan = d.estado.datos["planes"]["contratos"]
    assert plan["filas_descartadas_por_anio"] == {"2018": 1, "2026": 1} and plan["archivos_en_rango"] == 2
    chequeo = d.estado.datos["chequeos"]["contratos"]
    assert chequeo["total"] == 12 and chequeo["repetidas_eliminadas"] == 1 and chequeo["ok"]
    assert d.estado.manifiesto[("contratos", "2024-07_local")]["error"].startswith("1 fila(s) repetida(s)")


def test_archivo_ausente_deja_la_unidad_fallida_y_no_sigue(carpeta_pequena, tmp_path, monkeypatch):
    (carpeta_pequena / "2024-08.parquet").unlink()
    servidor = Servidor()
    instalar(monkeypatch, servidor)
    codigo, comp = lanzar(["--contratos-local", str(carpeta_pequena), "--destino", str(tmp_path / "b")])
    assert codigo == 2 and comp["ok"] is False and comp["motivo"] == "contratos_pendientes"
    assert [f["clave"] for f in comp["faltantes"]["contratos"]] == ["2024-08_local"]
    assert servidor.registro == []                                       # sin Procesos ni Modificaciones


def test_total_distinto_del_esperado_falla(carpeta_pequena, tmp_path, monkeypatch):
    monkeypatch.setattr(descarga, "CONTRATOS_LOCAL_ESPERADOS", 13)
    servidor = Servidor()
    instalar(monkeypatch, servidor)
    codigo, comp = lanzar(["--contratos-local", str(carpeta_pequena), "--destino", str(tmp_path / "b")])
    assert codigo == 1 and comp["ok"] is False and comp["motivo"] == "chequeo_contratos_fallo"
    assert comp["chequeos"]["contratos"]["total"] == 12 and servidor.registro == []


def test_corrida_pequena_completa_y_conteo_distinto_del_servidor(carpeta_pequena, tmp_path, monkeypatch):
    servidor = Servidor()
    instalar(monkeypatch, servidor)
    argv = ["--contratos-local", str(carpeta_pequena), "--destino", str(tmp_path / "b")]
    codigo, comp = lanzar(argv)
    assert codigo == 0 and comp["ok"] is True and comp["modo"] == "contratos_local"
    assert comp["corte"] == descarga.CORTE_LOCAL and comp["digito"] is None
    assert {k: v["coinciden"] for k, v in comp["conteo_lotes"].items()} == {"procesos": 1, "modificaciones": 1}
    assert comp["unidades"]["contratos"] == {"planificadas": 2, "hecho": 2, "pendiente": 0, "fallido": 0}
    assert not [r for r in servidor.registro if r["dataset"] == "contratos"]   # Contratos jamas va al servidor
    # un servidor que cuenta una fila de mas por lote: otra carpeta, sale con 1 y lista las diferencias
    servidor2 = Servidor()
    servidor2.sumar_al_conteo = 1
    instalar(monkeypatch, servidor2)
    codigo, comp = lanzar(["--contratos-local", str(carpeta_pequena), "--destino", str(tmp_path / "c")])
    assert codigo == 1 and comp["ok"] is False and comp["motivo"] == "conteo_lotes_no_cuadra"
    assert len(comp["conteo_lotes"]["procesos"]["diferencias"]) == 1
    assert comp["conteo_lotes"]["procesos"]["diferencias"][0]["servidor"] == comp["conteo_lotes"]["procesos"][
        "diferencias"][0]["descargado"] + 1


def test_parametros_de_modo_local(monkeypatch, tmp_path):
    capturado = {}

    def parar(args):
        capturado["args"] = args
        raise RuntimeError("fin")
    monkeypatch.setattr(descarga, "configurar_log", lambda destino: None)
    monkeypatch.setattr(descarga, "Descarga", parar)
    with pytest.raises(RuntimeError):
        descarga.main(["--contratos-local"])
    a = capturado["args"]
    assert a.destino == str(descarga.DESTINO_LOCAL) and descarga.DESTINO_LOCAL.name == "f2_b"
    assert a.corte == "2026-10-14T22:53:00" and a.digito is None and a.contratos_local == str(descarga.CONTRATOS_LOCAL)
    with pytest.raises(SystemExit):
        descarga.main(["--contratos-local", "--digito", "4"])          # sin filtro por digito
    with pytest.raises(SystemExit):
        descarga.main(["--contratos-local", str(tmp_path / "no_existe")])


# ---------------- conversion del corte (B1) ----------------
def test_conversion_del_corte(tmp_path):
    """Los campos de fecha vienen a medianoche (calendar_date): el corte local cuenta solo como fecha <= 2026-10-14."""
    corte = descarga.CORTE_LOCAL
    assert corte == "2026-10-14T22:53:00"
    d = descarga.Descarga(argumentos(tmp_path / "b", contratos_local=str(tmp_path), corte=corte))
    where = d.where_lote("modificaciones", ["CO1.PCCNTR.1", "CO1.PCCNTR.2"])
    assert where == ("id_contrato in ('CO1.PCCNTR.1','CO1.PCCNTR.2') AND estado_modificacion = 'Publicado' AND "
                     "dias_extendidos > 0 AND fecha_version <= '2026-10-14T22:53:00'")
    assert "2026-10-15T03:53:00" not in where
    fechas = pd.to_datetime(["2026-10-13T00:00:00.000", "2026-10-14T00:00:00.000", "2026-10-15T00:00:00.000"])
    assert list(fechas <= pd.Timestamp(corte)) == [True, True, False]
    # la hora convertida a UTC dejaria entrar toda la fecha 2026-10-15, posterior a la foto de Contratos
    assert list(fechas <= pd.Timestamp("2026-10-15T03:53:00")) == [True, True, True]


# ---------------- los 106 archivos reales ----------------
@pytest.fixture(scope="module")
def base_b(tmp_path_factory):
    """Copia local de los 106 archivos a unidades de F2 (una sola vez por modulo)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SOCRATA_APP_TOKEN", TOKEN)
        destino = tmp_path_factory.mktemp("f2_b")
        d = descarga.Descarga(argumentos(destino, contratos_local=str(LOCAL), corte=descarga.CORTE_LOCAL))
        unidades = d.unidades_contratos()
        d.copiar_contratos_local(unidades)
        ok = d.chequear_contratos_local(unidades)
        return d, unidades, ok, destino


@requiere_local
def test_lectura_de_los_106_archivos_y_conteo(base_b):
    d, unidades, ok, destino = base_b
    assert hay_local and len(list(LOCAL.glob("*.parquet"))) == 106
    assert len(unidades) == 84 and ok is True
    chequeo = d.estado.datos["chequeos"]["contratos"]
    assert chequeo["total"] == 167_997 == descarga.CONTRATOS_LOCAL_ESPERADOS
    assert chequeo["repetidas_eliminadas"] == 3 and chequeo["duplicados_id_contrato"] == 0
    assert chequeo["filas_fuera_de_unidad"] == 0
    assert {f["anio"]: f["contratos"] for f in chequeo["por_anio"]} == {
        2019: 24_000, 2020: 24_000, 2021: 24_000, 2022: 23_998, 2023: 24_000, 2024: 23_999, 2025: 24_000}
    plan = d.estado.datos["planes"]["contratos"]
    assert plan["filas_descartadas_por_anio"] == {"2018": 24_000, "2026": 20_000}
    assert all(f["estado"] == "hecho" for f in d.estado.filas("contratos"))
    ids = pd.concat([pd.read_parquet(u["ruta"], columns=["id_contrato"]) for u in unidades])["id_contrato"]
    assert len(ids) == 167_997 and ids.is_unique
    assert set(ids.str[-1]) == set("0123456789")                          # sin filtro por digito
    # los mismos contratos de la base v0 (python -m src.base), si esta construida
    v0 = config.DATA_PROCESSED / "base_contratos.parquet"
    if v0.exists():
        assert set(pd.read_parquet(v0, columns=["id_contrato"])["id_contrato"]) == set(ids)


@requiere_local
def test_formato_igual_al_de_f2_normal(base_b):
    d, unidades, _, destino = base_b
    ruta = destino / "contratos" / "anio=2019" / "mes=01_local.parquet"
    assert ruta.exists() and re.fullmatch(r"mes=\d\d_local\.parquet", ruta.name)
    df = pd.read_parquet(ruta)
    assert list(df.columns) == descarga.COLS["contratos"] and len(df.columns) == 39
    assert not set(df.columns) & config.PII
    assert len(df) == 2000
    # mismo esquema que una unidad de F2 normal (la prueba del paso 5), si existe en este equipo
    a = config.DATA_RAW / "f1_verificacion" / "paso5" / "rango_completa" / "contratos" / "anio=2019" / "mes=01_dig=4.parquet"
    if a.exists():
        assert pd.read_parquet(a).dtypes.equals(df.dtypes)


@requiere_local
def test_plan_de_lotes(base_b):
    d, unidades, _, _ = base_b
    ids_contrato = pd.concat([pd.read_parquet(u["ruta"], columns=["id_contrato", "proceso_de_compra"]) for u in unidades])
    portafolios = ids_contrato["proceso_de_compra"].dropna()
    esperados = {"procesos": portafolios[portafolios.str.strip() != ""].nunique(), "modificaciones": 167_997}
    for dataset, n_ids in esperados.items():
        lotes = d.lotes(dataset)
        todos = [i for l in lotes for i in l["ids"]]
        assert len(todos) == len(set(todos)) == n_ids                       # cada ID en un solo lote
        assert all(1 <= len(l["ids"]) <= descarga.TAMANO_LOTE for l in lotes)
        assert max(d.largo_url(dataset, l["ids"]) for l in lotes) <= descarga.MAX_URL
        plan = d.estado.datos["planes"][dataset]
        assert plan["ids"] == n_ids and plan["lotes"] == len(lotes)
        assert len(lotes) == math.ceil(n_ids / descarga.TAMANO_LOTE) + plan["partidos_por_url"]
        assert [l["clave"] for l in lotes] == [l["clave"] for l in d.lotes(dataset)]      # mismos lotes al reanudar
        assert len({l["clave"] for l in lotes}) == len(lotes)
    assert len(d.lotes("modificaciones")) >= 336


@requiere_local
def test_corrida_completa_con_corte_duro_corte_por_hora_y_reanudacion(tmp_path, monkeypatch):
    """B con servidor falso (2022-2023, 48.000 contratos): Ctrl+C dentro de Procesos, --hasta dentro de Modificaciones,
    reanudacion y comparacion con una corrida sin cortes. El rango completo (84 meses, 167.997) tambien se corrio una vez."""
    monkeypatch.setattr(descarga, "DESDE", "2022-01")
    monkeypatch.setattr(descarga, "HASTA", "2023-12")
    monkeypatch.setattr(descarga, "CONTRATOS_LOCAL_ESPERADOS", 23_998 + 24_000)
    servidor = Servidor()
    instalar(monkeypatch, servidor)
    argv = ["--contratos-local", "--destino", str(tmp_path / "cortada")]

    servidor.interrumpir_en = 30                                        # corte duro dentro de Procesos
    assert descarga.main(argv) == 130
    est = json.loads((tmp_path / "cortada" / "_estado.json").read_text(encoding="utf-8"))
    manifiesto = pd.read_csv(tmp_path / "cortada" / "_manifiesto.csv", dtype=str, keep_default_na=False)
    procesos = manifiesto[manifiesto.dataset == "procesos"]
    assert (procesos.estado == "hecho").sum() == 29 and (manifiesto.dataset == "contratos").sum() == 24
    assert not ((manifiesto.estado == "pendiente") & (manifiesto.inicio != "")).any()      # nada a medias
    assert est["corte"] == "2026-10-14T22:53:00"
    servidor.interrumpir_en = None
    hechos_1 = set(zip(manifiesto[manifiesto.estado == "hecho"].dataset, manifiesto[manifiesto.estado == "hecho"].clave))
    pedidas_1 = len(servidor.registro)

    original = descarga.Descarga.revisar_tope                          # --hasta: se alcanza dentro de Modificaciones
    llamadas = {"n": 0}

    def tope(self):
        llamadas["n"] += 1
        if self.estado.filas("modificaciones") and sum(f["estado"] == "hecho" for f in self.estado.filas(
                "modificaciones")) >= 20:
            raise descarga.TopeTiempo()
        return original(self)
    monkeypatch.setattr(descarga.Descarga, "revisar_tope", tope)
    codigo, comp = lanzar(argv)
    assert codigo == 4 and comp["motivo"] == "hora_limite" and comp["ok"] is False
    assert comp["unidades"]["modificaciones"]["hecho"] == 20 and comp["unidades"]["procesos"]["pendiente"] == 0
    monkeypatch.setattr(descarga.Descarga, "revisar_tope", original)
    manifiesto = pd.read_csv(tmp_path / "cortada" / "_manifiesto.csv", dtype=str, keep_default_na=False)
    hechos_2 = set(zip(manifiesto[manifiesto.estado == "hecho"].dataset, manifiesto[manifiesto.estado == "hecho"].clave))
    sha_antes = dict(zip(manifiesto.clave, manifiesto.sha256))
    pedidas_2 = len(servidor.registro)

    codigo, comp = lanzar(argv)                                         # reanudacion hasta el final
    assert codigo == 0 and comp["ok"] is True and comp["motivo"] == "completo"
    nuevas = servidor.registro[pedidas_2:]
    datos_nuevos = {(r["dataset"], r["clave"]) for r in nuevas if r["tipo"] == "datos"}
    assert not datos_nuevos & hechos_2                                  # ninguna unidad hecha se volvio a pedir
    assert not {(r["dataset"], r["clave"]) for r in servidor.registro[pedidas_1:pedidas_2]
                if r["tipo"] == "datos"} & hechos_1
    manifiesto = pd.read_csv(tmp_path / "cortada" / "_manifiesto.csv", dtype=str, keep_default_na=False)
    assert all(sha_antes[c] == s for c, s in zip(manifiesto.clave, manifiesto.sha256) if c in sha_antes and sha_antes[c])
    assert comp["chequeos"]["contratos"]["total"] == 47_998
    assert comp["unidades"]["contratos"]["hecho"] == 24
    for dataset in ("procesos", "modificaciones"):
        c = comp["conteo_lotes"][dataset]
        assert c["lotes"] > 80 and c["coinciden"] == c["lotes"] and not c["diferencias"] and c["sin_verificar"] == 0
        conteos = servidor.claves(dataset, "conteo")
        assert len(conteos) == len(set(conteos)) == c["lotes"]            # cada lote se conto una sola vez
    assert not [r for r in servidor.registro if r["dataset"] == "contratos"]
    assert all("fecha_version <= '2026-10-14T22:53:00'" in r["where"] for r in servidor.registro
               if r["dataset"] == "modificaciones")

    # comparacion fila por fila con una corrida sin cortes
    servidor_ref = Servidor()
    instalar(monkeypatch, servidor_ref)
    codigo, comp_ref = lanzar(["--contratos-local", "--destino", str(tmp_path / "referencia")])
    assert codigo == 0 and comp_ref["ok"] is True
    for carpeta in ("contratos", "procesos", "modificaciones"):
        rutas = sorted(p.relative_to(tmp_path / "cortada") for p in (tmp_path / "cortada" / carpeta).rglob("*.parquet"))
        assert rutas == sorted(p.relative_to(tmp_path / "referencia")
                               for p in (tmp_path / "referencia" / carpeta).rglob("*.parquet"))
        for r in rutas:
            pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "cortada" / r),
                                          pd.read_parquet(tmp_path / "referencia" / r))
    assert not list((tmp_path / "cortada").rglob("*.tmp"))


# ---------------- el modo por defecto no cambia ----------------
def test_por_defecto_argumentos_y_estado(monkeypatch, tmp_path):
    Real = descarga.Descarga
    capturado = {}

    def parar(args):
        capturado["args"] = args
        raise RuntimeError("fin")
    monkeypatch.setattr(descarga, "configurar_log", lambda destino: None)
    monkeypatch.setattr(descarga, "Descarga", parar)
    with pytest.raises(RuntimeError):
        descarga.main([])
    a = capturado["args"]
    assert (a.digito, a.destino, a.corte, a.contratos_local, a.mes) == ("4", str(config.DATA_RAW / "f2"), None, None, None)
    assert descarga.DESTINO == config.DATA_RAW / "f2" and descarga.DIGITO == "4"
    assert (descarga.DESDE, descarga.HASTA, descarga.TAMANO_LOTE, descarga.MAX_URL) == ("2019-01", "2025-12", 500, 15_000)
    assert [len(descarga.COLS[k]) for k in ("contratos", "procesos", "modificaciones")] == [39, 20, 8]

    d = Real(argumentos(tmp_path / "a", digito="4", corte="2026-10-26T07:30:00"))
    assert d.local is None and d.digito == "4"
    assert set(d.estado.datos["parametros"]) == {"digito", "desde", "hasta", "tamano_lote", "max_url", "limite_contratos",
                                                 "limite_lotes", "version"}          # sin claves nuevas: A reanuda igual
    u = d.unidades_contratos()
    assert len(u) == 84 and u[0]["clave"] == "2019-01_dig=4"
    assert u[0]["ruta"] == (tmp_path / "a").resolve() / "contratos" / "anio=2019" / "mes=01_dig=4.parquet"
    assert u[0]["where"] == ("fecha_de_firma >= '2019-01-01T00:00:00' AND fecha_de_firma < '2019-02-01T00:00:00' "
                             "AND id_contrato like '%4'")
    assert d.where_lote("procesos", ["a"]) == "id_del_portafolio in ('a')"
    assert d.where_lote("modificaciones", ["a"]) == ("id_contrato in ('a') AND estado_modificacion = 'Publicado' AND "
                                                     "dias_extendidos > 0 AND fecha_version <= '2026-10-26T07:30:00'")
    # una carpeta con estado de F2 normal no admite reanudar en modo local (y al reves)
    with pytest.raises(SystemExit):
        Real(argumentos(tmp_path / "a", contratos_local=str(tmp_path), corte="2026-10-26T07:30:00"))


def test_por_defecto_corrida_con_servidor_falso(monkeypatch, tmp_path):
    contratos = [fila_contrato(i, "2024-08", f"CO1.BDOS.{i // 2}") | {"id_contrato": f"CO1.PCCNTR.{i}4"}
                 for i in range(5)]
    servidor = Servidor(contratos)
    instalar(monkeypatch, servidor)
    codigo, comp = lanzar(["--mes", "2024-08", "--destino", str(tmp_path / "a"), "--corte", "2026-10-26T07:30:00"])
    assert codigo == 0 and comp["ok"] is True and comp["motivo"] == "completo" and comp["digito"] == "4"
    assert "modo" not in comp and "conteo_lotes" not in comp               # completitud.json igual que antes
    assert [(r["dataset"], r["tipo"]) for r in servidor.registro] == [
        ("contratos", "datos"), ("contratos", "conteo_mes"), ("procesos", "datos"), ("modificaciones", "datos")]
    primera = servidor.registro[0]["params"]
    assert primera["$where"] == ("fecha_de_firma >= '2024-08-01T00:00:00' AND fecha_de_firma < '2024-09-01T00:00:00' "
                                 "AND id_contrato like '%4'")
    assert (primera["$order"], primera["$limit"], primera["$offset"]) == (":id", 10_000, 0)
    assert primera["$select"] == ",".join(descarga.COLS["contratos"])
    assert servidor.registro[3]["params"]["$limit"] == 50_000
    assert (tmp_path / "a" / "contratos" / "anio=2024" / "mes=08_dig=4.parquet").exists()
