"""Descarga F2 de SECOP II: Contratos, Procesos y Modificaciones de la muestra por ultimo digito (PLAN.md 4.6).

Combinacion aprobada en el paso 4 de la F1 (.agents/tasks/f1/paso4_estimacion_f2.md, seccion 8):
- Contratos (jbjy-vk9h): una unidad por mes de firma x digito, con fecha_de_firma >= 'AAAA-MM-01T00:00:00' AND
  fecha_de_firma < '<mes siguiente>' AND id_contrato like '%D', en paginas de 10.000 filas ($order=:id, $offset
  mientras la pagina venga llena). Columnas de la seccion 4.2 del plan.
- Procesos (p6dx-8zbt): id_del_portafolio in (<500 IDs>), con los proceso_de_compra unicos de los Contratos
  descargados; paginas de 50.000 ($order=:id).
- Modificaciones (u8cx-r425): id_contrato in (<500 IDs>) AND estado_modificacion = 'Publicado' AND
  dias_extendidos > 0 AND fecha_version <= '<corte>', con los id_contrato descargados; paginas de 50.000. Columnas
  minimas del paso 3: sin autor_version, sustituto_version, proposito_modificacion ni descripcion.
- Si la URL preparada de un lote pasa de 15.000 caracteres, el lote se parte en dos (el servidor da 414 hacia 16 KiB).
- Una sola fecha de corte para los tres conjuntos (la hora de arranque, o --corte), guardada en _estado.json, en cada
  fila del manifiesto y en completitud.json.

Toda peticion pasa por src/cliente_socrata.py: 1 por segundo de inicio a inicio, en secuencia, App Token solo en el
header X-App-Token, timeout de 30 s (120 s para el conteo de control), 3 intentos con espera exponencial (5 s, 10 s)
ante 5xx, timeouts y 429, respetando Retry-After. 403, o 429 en el ultimo intento, detienen todo (Throttled).

Reanudacion (estado en disco, en --destino):
- Un parquet por unidad (contratos/anio=AAAA/mes=MM_dig=D.parquet, procesos/lote_<sha1>.parquet,
  modificaciones/lote_<sha1>.parquet), escrito en .tmp y renombrado al terminar: si existe, esta hecho.
- Los lotes se nombran por el sha1 de sus IDs ordenados, asi que al reanudar salen los mismos lotes.
- _manifiesto.csv (escritura atomica) tiene una fila por unidad del plan con estado pendiente / hecho / fallido. Una
  unidad pendiente con `inicio` y sin parquet quedo a medias: se rehace entera. Una hecha no se vuelve a pedir.
- Se puede matar el proceso en cualquier momento (Ctrl+C, Stop-Process): el estado queda consistente.

Chequeo de completitud (completitud.json en --destino, se escribe al final de cada corrida, tambien si se corta):
- Contratos: filas por mes contra un count(*) del servidor del mismo filtro, pedido al terminar Contratos;
  id_contrato unico; filas dentro de su mes y digito; con anios completos, filas por anio x digito contra el conteo
  del paso 4 y contra el 10 % de la poblacion (+-2 %).
- Procesos y Modificaciones: todos los lotes del plan hechos.
- `ok` es true solo si todo eso se cumple; `faltantes` lista las unidades no hechas de cada conjunto.

Hora limite (--hasta AAAA-MM-DDTHH:MM, hora local): se revisa antes de cada peticion de datos. Al llegar, la unidad en
curso se descarta y vuelve a pendiente (no queda nada a medias), se escribe el manifiesto y completitud.json
(motivo "hora_limite") y sale con codigo 4. Reanudar con el mismo comando (con otra --hasta) sigue donde quedo.

Modo de Contratos local (--contratos-local [CARPETA], base B; decision C del usuario, Contratos truncado el 25-oct):
- Contratos no se descarga: se lee de los archivos AAAA-MM.parquet de la exploracion (foto del 14-oct de 2026, 106
  archivos de 2.000 filas en 00_exploracion/data/raw/contratos_chunks/). Se conservan 2019-01 a 2025-12 (84 meses; se
  descartan 2018 y 2026), se quitan las filas exactamente repetidas del servidor (3) y se escribe una unidad por mes
  con las mismas columnas (COLS["contratos"], 39) y el mismo formato que en F2:
  contratos/anio=AAAA/mes=MM_local.parquet. Sin filtro por digito. Total esperado: 167.997 contratos, los mismos de la
  base v0 (python -m src.base).
- Procesos y Modificaciones se descargan como siempre (lotes de 500 IDs, URL <= 15.000 caracteres), con corte por
  defecto 2026-10-14T22:53:00 (hora de Colombia, sin zona). Zona horaria (B1): los campos de fecha de Modificaciones
  y Procesos son calendar_date (flotantes) y todos sus valores vienen a medianoche (T00:00:00.000), asi que el corte
  solo cuenta como fecha: `fecha_version <= '2026-10-14T22:53:00'` equivale a fecha <= 2026-10-14. No se convierte a
  UTC: 2026-10-15T03:53:00 incluiria toda la fecha 2026-10-15, posterior a la foto de Contratos.
- Destino por defecto data/raw/f2_b/. No aplican los chequeos de tamano (10 %) ni de digito. Completitud de B: 167.997
  contratos unicos, cada mes dentro de su unidad, todos los lotes hechos y, por lote, count(*) del servidor (mismo
  $where) igual a las filas descargadas (se pide al terminar las descargas; se guarda por lote y se reanuda).
  `ok` en completitud.json exige todo eso. Codigos de salida: los mismos (1 si algun lote no coincide con el servidor,
  2 si quedan conteos sin verificar).

Uso (desde final-project/, con .venv/Scripts/python.exe):
  python -m src.descarga --hasta 2026-10-26T06:00          # F2: 2019-2025, digito 4, en data/raw/f2/; reanudar igual
  python -m src.descarga --contratos-local --hasta 2026-10-26T11:30   # base B en data/raw/f2_b/; reanudar igual
  python -m src.descarga --mes 2024-06 --destino data/raw/f1_verificacion/paso5/prueba
  python -m src.descarga --conjuntos contratos,procesos    # deja Modificaciones pendiente (codigo 2)
Codigos de salida: 0 todo hecho y chequeo OK; 1 fallo el chequeo de Contratos, una consulta invalida u otro error;
2 quedan unidades pendientes o fallidas (reanudar con el mismo comando); 3 el portal limito el uso (403/429);
4 se alcanzo --hasta; 130 interrumpido. Solo el 0 corresponde a completitud.json con "ok": true.
"""
import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
import requests

from . import config
from . import features as F
from .cliente_socrata import ClienteSocrata, Throttled, leer_token

VERSION = "1"
DIGITO = "4"                       # aprobado en el checkpoint del paso 4 de la F1
DESDE, HASTA = "2019-01", "2025-12"
DESTINO = config.DATA_RAW / "f2"
DESTINO_LOCAL = config.DATA_RAW / "f2_b"                                              # base B (--contratos-local)
CONTRATOS_LOCAL = config.RAIZ / "00_exploracion" / "data" / "raw" / "contratos_chunks"
CONTRATOS_LOCAL_ESPERADOS = 167_997   # 2019-2025: 7 anios x 24.000 - 3 filas repetidas del servidor (base v0)
CORTE_LOCAL = "2026-10-14T22:53:00"   # foto de Contratos del 14-oct, hora de Colombia sin zona (B1: no se convierte a UTC)
TAMANO_LOTE = 500
MAX_URL = 15_000
LIMITE_CONTRATOS = 10_000
LIMITE_LOTES = 50_000
CONJUNTOS = ("contratos", "procesos", "modificaciones")
PENDIENTE, HECHO, FALLIDO = "pendiente", "hecho", "fallido"

COLS = {
    # PLAN.md 4.2 (sin PII)
    "contratos": [
        "id_contrato", "proceso_de_compra", "referencia_del_contrato", "fecha_de_firma",
        "nombre_entidad", "nit_entidad", "departamento", "ciudad", "orden", "sector", "rama", "entidad_centralizada",
        "tipo_de_contrato", "modalidad_de_contratacion", "justificacion_modalidad_de",
        "codigo_de_categoria_principal", "objeto_del_contrato", "es_grupo", "es_pyme", "tipodocproveedor",
        "habilita_pago_adelantado", "valor_de_pago_adelantado", "origen_de_los_recursos", "destino_gasto",
        "espostconflicto", "documentos_tipo", "el_contrato_puede_ser_prorrogado",
        "presupuesto_general_de_la_nacion_pgn", "sistema_general_de_participaciones", "sistema_general_de_regal_as",
        "recursos_propios_alcald_as_gobernaciones_y_resguardos_ind_genas_", "recursos_de_credito", "recursos_propios",
        "fecha_de_inicio_del_contrato", "duraci_n_del_contrato", "dias_adicionados",
        "fecha_de_fin_del_contrato", "estado_contrato", "valor_del_contrato"],
    "procesos": [
        "id_del_portafolio", "id_del_proceso", "adjudicado", "fecha_adjudicacion", "valor_total_adjudicacion",
        "fecha_de_publicacion_del", "fecha_de_recepcion_de", "fecha_de_apertura_efectiva",
        "precio_base", "duracion", "unidad_de_duracion", "numero_de_lotes", "subtipo_de_contrato",
        "proveedores_invitados", "proveedores_con_invitacion", "proveedores_que_manifestaron",
        "respuestas_al_procedimiento", "respuestas_externas", "conteo_de_respuestas_a_ofertas",
        "proveedores_unicos_con"],
    # columnas minimas del paso 3 de la F1
    "modificaciones": [
        "id_contrato", "identificador_modificacion", "version_anterior_del_contrato", "fecha_version",
        "estado_modificacion", "dias_extendidos", "fecha_de_aprobacion", "fecha_fin_contrato"],
}
FILTRO_MODIF = "estado_modificacion = 'Publicado' AND dias_extendidos > 0 AND fecha_version <= '{corte}'"
COLUMNA_IDS = {"procesos": "proceso_de_compra", "modificaciones": "id_contrato"}   # columna de Contratos que alimenta
COLUMNA_FILTRO = {"procesos": "id_del_portafolio", "modificaciones": "id_contrato"}

for _c in COLS.values():
    assert not set(_c) & config.PII
assert not {"autor_version", "sustituto_version", "proposito_modificacion", "descripcion"} & set(COLS["modificaciones"])
assert set(F.COLS_CONTRATO) <= set(COLS["contratos"])
assert {c.removeprefix("p_") for c in F.COLS_PROCESO} <= set(COLS["procesos"])

# Conteos del paso 4 de la F1 (datos.gov.co, 2026-10-24 11:22): contratos por anio de firma del digito 4 y poblacion.
CONTEO_PASO4 = {"4": {2019: 14407, 2020: 36107, 2021: 56469, 2022: 71095, 2023: 83764, 2024: 88548, 2025: 102125}}
POBLACION_PASO4 = {2019: 142090, 2020: 356623, 2021: 560441, 2022: 707765, 2023: 835462, 2024: 881978,
                   2025: 1017075}
# Tolerancias (Contratos se actualiza a diario): contra el count(*) pedido al terminar Contratos, max(5, 0,1 %) por mes;
# contra el conteo del paso 4, max(10, 0,1 %) por anio. Un mes faltante (>= ~200 filas) o una pagina de 10.000 las
# superan. Contra el 10 % de la poblacion, +-2 % relativo (PLAN.md, F2).
TOL_MES = (5, 0.001)
TOL_PASO4 = (10, 0.001)
TOL_POBLACION = 0.02

log = logging.getLogger("descarga")


class TopeTiempo(Exception):
    """Se alcanzo la hora de --hasta: se detiene antes de la siguiente peticion de datos."""


def ahora():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def tolerancia(n, tol):
    minimo, relativa = tol
    return max(minimo, math.ceil(relativa * n))


def in_list(valores):
    return ",".join("'" + str(v).replace("'", "''") + "'" for v in valores)


def sha1_ids(ids):
    return hashlib.sha1(",".join(ids).encode("utf-8")).hexdigest()


def escribir_atomico(ruta, contenido):
    """Escribe en <ruta>.tmp y renombra. Reintenta el renombre si OneDrive o un lector tiene el archivo abierto."""
    tmp = ruta.with_name(ruta.name + ".tmp")
    datos = contenido.encode("utf-8") if isinstance(contenido, str) else contenido
    tmp.write_bytes(datos)
    for intento in range(10):
        try:
            os.replace(tmp, ruta)
            return
        except PermissionError:
            if intento == 9:
                raise
            time.sleep(0.5)


def guardar_parquet(filas, cols, ruta):
    """Guarda las filas (texto, como las devuelve la API) en un parquet atomico; devuelve (filas, bytes, sha256)."""
    df = pd.DataFrame(filas).reindex(columns=cols) if filas else pd.DataFrame(columns=cols)
    df = df.astype("string")
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    contenido = buf.getvalue()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    escribir_atomico(ruta, contenido)
    return len(df), len(contenido), hashlib.sha256(contenido).hexdigest()


def meses(desde, hasta):
    """Lista de (anio, mes, inicio, fin) desde 'AAAA-MM' hasta 'AAAA-MM', ambos incluidos."""
    ini, fin = pd.Period(desde, "M"), pd.Period(hasta, "M")
    if fin < ini:
        raise ValueError(f"rango de meses vacio: {desde} a {hasta}")
    return [(p.year, p.month, p.start_time, (p + 1).start_time) for p in pd.period_range(ini, fin, freq="M")]


class Estado:
    """Estado persistente en --destino: _estado.json (parametros, corte, planes, chequeos) y _manifiesto.csv."""

    CAMPOS = ["dataset", "clave", "estado", "archivo", "n_ids", "filas", "paginas", "bytes", "segundos", "inicio",
              "fin", "sha256", "corte", "error"]

    def __init__(self, destino, parametros, corte_pedido):
        self.destino = destino
        destino.mkdir(parents=True, exist_ok=True)
        self.ruta_estado = destino / "_estado.json"
        self.ruta_manifiesto = destino / "_manifiesto.csv"
        if self.ruta_estado.exists():
            self.datos = json.loads(self.ruta_estado.read_text(encoding="utf-8"))
            previos = self.datos["parametros"]
            distintos = {k: (previos.get(k), v) for k, v in parametros.items() if previos.get(k) != v}
            if distintos:
                raise SystemExit(f"{destino} ya tiene una descarga con otros parametros {distintos}; "
                                 "use el mismo comando o otro --destino")
            if corte_pedido and corte_pedido != self.datos["corte"]:
                raise SystemExit(f"{destino} ya tiene el corte {self.datos['corte']}; no se puede cambiar a "
                                 f"{corte_pedido}")
        else:
            self.datos = {"version": VERSION, "parametros": parametros, "corte": corte_pedido or ahora(),
                          "creado": ahora(), "planes": {}, "chequeos": {}}
            self.guardar()
        self.manifiesto = {}
        if self.ruta_manifiesto.exists():
            with self.ruta_manifiesto.open(newline="", encoding="utf-8") as f:
                for fila in csv.DictReader(f):
                    self.manifiesto[(fila["dataset"], fila["clave"])] = fila

    @property
    def corte(self):
        return self.datos["corte"]

    def guardar(self):
        escribir_atomico(self.ruta_estado, json.dumps(self.datos, indent=2, ensure_ascii=True))

    def guardar_manifiesto(self):
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=self.CAMPOS, lineterminator="\n")
        w.writeheader()
        w.writerows(self.manifiesto.values())
        escribir_atomico(self.ruta_manifiesto, buf.getvalue())

    def marcar(self, dataset, clave, guardar=True, **campos):
        fila = self.manifiesto.setdefault((dataset, clave), {c: "" for c in self.CAMPOS})
        fila.update(dataset=dataset, clave=clave, corte=self.corte,
                    **{k: "" if v is None else v for k, v in campos.items()})
        if guardar:
            self.guardar_manifiesto()

    def filas(self, dataset):
        return [f for (d, _), f in self.manifiesto.items() if d == dataset]

    def planificar(self, dataset, unidades):
        """Registra cada unidad del plan (pendiente si es nueva), reconcilia el manifiesto con los archivos y devuelve
        las unidades por pedir. Si el parquet existe pero el manifiesto no lo marca hecho (el proceso murio entre el
        renombre y el manifiesto), se marca hecho desde el archivo, sin volver a pedirlo."""
        claves = {u["clave"] for u in unidades}
        viejas = [k for k in self.manifiesto if k[0] == dataset and k[1] not in claves]
        for k in viejas:
            del self.manifiesto[k]
        if viejas:
            log.warning(f"[{dataset}] {len(viejas)} unidades del manifiesto ya no estan en el plan; se quitan")
        por_pedir = []
        for u in unidades:
            clave, ruta = u["clave"], u["ruta"]
            archivo = ruta.relative_to(self.destino).as_posix()
            fila = self.manifiesto.get((dataset, clave))
            if fila is None:
                self.marcar(dataset, clave, guardar=False, estado=PENDIENTE, archivo=archivo,
                            n_ids=len(u["ids"]) if "ids" in u else "")
                fila = self.manifiesto[(dataset, clave)]
            if ruta.exists():
                if fila["estado"] != HECHO:
                    contenido = ruta.read_bytes()
                    n = len(pd.read_parquet(ruta, columns=[COLS[dataset][0]]))
                    self.marcar(dataset, clave, guardar=False, estado=HECHO, filas=n, bytes=len(contenido),
                                sha256=hashlib.sha256(contenido).hexdigest(), fin=ahora(),
                                error="reconciliado desde el archivo")
                    log.info(f"[{dataset}] {clave}: parquet presente, manifiesto reconciliado (no se vuelve a pedir)")
                else:
                    log.debug(f"[{dataset}] ya hecha, no se pide: {clave}")
                continue
            if fila["estado"] == HECHO:
                log.warning(f"[{dataset}] {clave}: el manifiesto la marca hecha pero falta el archivo; se rehace")
                fila["estado"] = PENDIENTE
            elif fila["estado"] == FALLIDO:
                log.info(f"[{dataset}] {clave}: fallo en una corrida anterior ({fila['error'][:80]}); se reintenta")
            elif fila["inicio"]:
                log.info(f"[{dataset}] {clave}: quedo a medias (inicio {fila['inicio']}); se rehace entera")
            por_pedir.append(u)
        self.guardar_manifiesto()
        return por_pedir


class Descarga:
    def __init__(self, args):
        self.local = Path(args.contratos_local).resolve() if getattr(args, "contratos_local", None) else None
        self.digito = None if self.local else args.digito
        self.desde, self.hasta = (args.mes, args.mes) if args.mes else (DESDE, HASTA)
        self.destino = Path(args.destino).resolve()
        parametros = {"digito": self.digito, "desde": self.desde, "hasta": self.hasta, "tamano_lote": TAMANO_LOTE,
                      "max_url": MAX_URL, "limite_contratos": LIMITE_CONTRATOS, "limite_lotes": LIMITE_LOTES,
                      "version": VERSION}
        if self.local:   # solo en modo local: el estado de F2 normal no cambia
            try:
                parametros["contratos_local"] = self.local.relative_to(config.RAIZ).as_posix()
            except ValueError:
                parametros["contratos_local"] = self.local.as_posix()
        self.estado = Estado(self.destino, parametros, args.corte)
        token = leer_token()
        self.cliente = ClienteSocrata(token=token, reintento_exponencial=True)
        self.agregado = ClienteSocrata(token=token, timeout=120, reintento_exponencial=True)   # count(*) de control
        del token
        self.hora_limite = datetime.fromisoformat(args.hora_limite) if args.hora_limite else None
        self.motivo = ""   # por que termino la corrida; va a completitud.json
        self.corrida = ahora()
        self.t0, self.t0_reloj = time.perf_counter(), datetime.now()
        self._volcados = {id(self.cliente): 0, id(self.agregado): 0}
        self._ultimo_inicio = None
        self._t_fase = 0.0
        self.actual = ("-", "-")   # unidad en curso, para el registro de peticiones si se interrumpe
        self.bajadas = {d: 0 for d in CONJUNTOS}
        self.ruta_peticiones = self.destino / "_peticiones.csv"
        self.ruta_completitud = self.destino / "completitud.json"

    # ---------------- registro ----------------
    def peticiones(self):
        return self.cliente.peticiones + self.agregado.peticiones

    def volcar_peticiones(self, dataset, clave):
        """Agrega a _peticiones.csv las peticiones nuevas (sin URL, sin IDs, sin token)."""
        nuevas = []
        for cli in (self.cliente, self.agregado):
            k = self._volcados[id(cli)]
            nuevas += cli.registro[k:]
            self._volcados[id(cli)] = len(cli.registro)
        if not nuevas:
            return
        nuevo = not self.ruta_peticiones.exists()
        with self.ruta_peticiones.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            if nuevo:
                w.writerow(["corrida", "dataset", "clave", "ruta", "intento", "hora_inicio", "separacion_s",
                            "segundos", "status"])
            for r in sorted(nuevas, key=lambda r: r["inicio"]):
                sep = "" if self._ultimo_inicio is None else round(r["inicio"] - self._ultimo_inicio, 3)
                self._ultimo_inicio = r["inicio"]
                hora = self.t0_reloj + pd.Timedelta(seconds=r["inicio"] - self.t0)
                w.writerow([self.corrida, dataset, clave, r["ruta"], r["intento"],
                            hora.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3], sep, round(r["segundos"], 3), r["status"]])

    def transcurrido(self):
        return time.perf_counter() - self.t0

    @staticmethod
    def hms(seg):
        seg = int(round(seg))
        return f"{seg // 3600:02d}:{seg % 3600 // 60:02d}:{seg % 60:02d}"

    def revisar_tope(self):
        if self.hora_limite is not None and datetime.now() >= self.hora_limite:
            raise TopeTiempo()

    # ---------------- unidades ----------------
    def unidades_contratos(self):
        out = []
        for anio, mes, ini, fin in meses(self.desde, self.hasta):
            if self.local:   # una unidad por mes, leida del archivo AAAA-MM.parquet de la exploracion
                out.append({"clave": f"{anio}-{mes:02d}_local", "fuente": self.local / f"{anio}-{mes:02d}.parquet",
                            "ruta": self.destino / "contratos" / f"anio={anio}" / f"mes={mes:02d}_local.parquet",
                            "anio": anio, "mes": f"{anio}-{mes:02d}", "ini": ini, "fin": fin})
                continue
            clave = f"{anio}-{mes:02d}_dig={self.digito}"
            ruta = self.destino / "contratos" / f"anio={anio}" / f"mes={mes:02d}_dig={self.digito}.parquet"
            where = (f"fecha_de_firma >= '{ini:%Y-%m-%dT%H:%M:%S}' AND fecha_de_firma < '{fin:%Y-%m-%dT%H:%M:%S}' "
                     f"AND id_contrato like '%{self.digito}'")
            out.append({"clave": clave, "ruta": ruta, "where": where, "anio": anio, "mes": f"{anio}-{mes:02d}",
                        "ini": ini, "fin": fin})
        return out

    def where_lote(self, dataset, ids):
        where = f"{COLUMNA_FILTRO[dataset]} in ({in_list(ids)})"
        if dataset == "modificaciones":
            where += " AND " + FILTRO_MODIF.format(corte=self.estado.corte)
        return where

    def params(self, dataset, where, limite, offset=0):
        return {"$select": ",".join(COLS[dataset]), "$where": where, "$order": ":id", "$limit": limite,
                "$offset": offset}

    def largo_url(self, dataset, ids):
        url = f"{self.cliente.base_url}/resource/{config.DATASETS[dataset]}.json"
        params = self.params(dataset, self.where_lote(dataset, ids), LIMITE_LOTES)
        return len(requests.Request("GET", url, params=params).prepare().url)

    def partir(self, dataset, ids):
        """Parte en dos (recursivo) mientras la URL preparada pase de MAX_URL."""
        if len(ids) > 1 and self.largo_url(dataset, ids) > MAX_URL:
            mitad = len(ids) // 2
            return self.partir(dataset, ids[:mitad]) + self.partir(dataset, ids[mitad:])
        return [ids]

    def lotes(self, dataset):
        """Lotes deterministas: IDs unicos y ordenados de los Contratos descargados, de a TAMANO_LOTE."""
        columna, valores = COLUMNA_IDS[dataset], set()
        for u in self.unidades_contratos():
            s = pd.read_parquet(u["ruta"], columns=[columna])[columna].dropna()
            valores.update(s[s.str.strip() != ""])
        ids = sorted(valores)
        lotes = []
        for i in range(0, len(ids), TAMANO_LOTE):
            lotes += self.partir(dataset, ids[i:i + TAMANO_LOTE])
        plan = {"ids": len(ids), "lotes": len(lotes),
                "partidos_por_url": len(lotes) - math.ceil(len(ids) / TAMANO_LOTE), "sha1_ids": sha1_ids(ids)}
        previo = self.estado.datos["planes"].get(dataset)
        if previo and previo["sha1_ids"] != plan["sha1_ids"]:
            log.warning(f"[{dataset}] el plan de lotes cambio desde la corrida anterior ({previo['ids']} -> "
                        f"{plan['ids']} IDs): los lotes viejos no se usan")
        self.estado.datos["planes"][dataset] = plan
        self.estado.guardar()
        log.info(f"[{dataset}] plan: {plan['ids']} IDs en {plan['lotes']} lotes de hasta {TAMANO_LOTE} "
                 f"({plan['partidos_por_url']} lotes extra por partir URL de mas de {MAX_URL} caracteres)")
        return [{"clave": "lote_" + sha1_ids(l)[:20], "ids": l,
                 "ruta": self.destino / dataset / ("lote_" + sha1_ids(l)[:20] + ".parquet")} for l in lotes]

    def bajar_unidad(self, dataset, unidad, where, limite, i, n):
        """Pide todas las paginas de una unidad y la guarda de una vez. Si algo falla, la unidad queda sin archivo."""
        clave, ruta = unidad["clave"], unidad["ruta"]
        self.actual = (dataset, clave)
        self.estado.marcar(dataset, clave, estado=PENDIENTE, inicio=ahora(), fin="", error="")
        t = time.perf_counter()
        filas, offset, paginas = [], 0, 0
        try:
            while True:
                if paginas:
                    self.revisar_tope()   # entre paginas de una misma unidad (Contratos llega a 3 paginas)
                datos = self.cliente.consultar(config.DATASETS[dataset], self.params(dataset, where, limite, offset))
                paginas += 1
                filas += datos
                if len(datos) < limite:
                    break
                offset += limite
        except (TopeTiempo, KeyboardInterrupt):
            # corte limpio: lo pedido de esta unidad se descarta y la unidad vuelve a pendiente, sin quedar a medias
            self.estado.marcar(dataset, clave, estado=PENDIENTE, inicio="", fin="",
                               error=f"cortada en la pagina {paginas + 1} por --hasta o Ctrl+C; se pide de nuevo")
            self.volcar_peticiones(dataset, clave)
            raise
        except Exception as e:  # noqa: BLE001 - fallo tras los reintentos: se registra y se sigue con la siguiente
            self.estado.marcar(dataset, clave, estado=FALLIDO, fin=ahora(), error=f"{type(e).__name__}: {e}"[:300])
            self.volcar_peticiones(dataset, clave)
            # 403/429 (Throttled) y 400 (ValueError) detienen todo; un JSON cortado se trata como fallo transitorio
            if isinstance(e, Throttled) or (isinstance(e, ValueError)
                                            and not isinstance(e, requests.exceptions.JSONDecodeError)):
                raise
            log.error(f"[{dataset}] {i}/{n} {clave}: FALLIDO {type(e).__name__}: {str(e)[:200]}")
            return
        n_filas, n_bytes, sha = guardar_parquet(filas, COLS[dataset], ruta)
        seg = time.perf_counter() - t
        self.estado.marcar(dataset, clave, estado=HECHO, filas=n_filas, paginas=paginas, bytes=n_bytes,
                           segundos=round(seg, 2), fin=ahora(), sha256=sha, error="")
        self.volcar_peticiones(dataset, clave)
        self.bajadas[dataset] += 1
        if dataset == "contratos" and self.estado.datos["chequeos"].pop("contratos", None):
            self.estado.guardar()   # Contratos cambio: el chequeo se rehace con un conteo nuevo del servidor
        eta = (self.transcurrido() - self._t_fase) / i * (n - i)
        ram = psutil.Process().memory_info().rss / 2 ** 20
        log.info(f"[{dataset}] {i}/{n} {clave}: {n_filas} filas, {paginas} pag, {seg:.1f} s | transcurrido "
                 f"{self.hms(self.transcurrido())} | ETA {dataset} {self.hms(eta)} | peticiones {self.peticiones()} | "
                 f"RAM {ram:.0f} MB")

    def recorrer(self, dataset, unidades, limite, where_de):
        por_pedir = self.estado.planificar(dataset, unidades)
        log.info(f"[{dataset}] {len(unidades)} unidades: {len(unidades) - len(por_pedir)} ya hechas, "
                 f"{len(por_pedir)} por pedir")
        self._t_fase = self.transcurrido()
        for i, u in enumerate(por_pedir, 1):
            self.revisar_tope()
            self.bajar_unidad(dataset, u, where_de(u), limite, i, len(por_pedir))

    def no_hechas(self, dataset):
        return [f for f in self.estado.filas(dataset) if f["estado"] != HECHO]

    # ---------------- chequeo de completitud ----------------
    def conteo_servidor(self, unidades):
        ini, fin = unidades[0]["ini"], unidades[-1]["fin"]
        self.actual = ("contratos", "conteo_servidor")
        datos = self.agregado.consultar(config.DATASETS["contratos"], {
            "$select": "date_trunc_ym(fecha_de_firma) as mes, count(*) as n",
            "$where": (f"fecha_de_firma >= '{ini:%Y-%m-%dT%H:%M:%S}' AND fecha_de_firma < '{fin:%Y-%m-%dT%H:%M:%S}' "
                       f"AND id_contrato like '%{self.digito}'"),
            "$group": "mes", "$order": "mes", "$limit": 1000})
        self.volcar_peticiones("contratos", "conteo_servidor")
        return {d["mes"][:7]: int(d["n"]) for d in datos}

    def chequear_contratos(self, unidades):
        """False si un mes no cuadra con el count(*) del servidor, si hay id_contrato repetidos o filas fuera de su
        unidad, o si un anio completo no cuadra con el conteo del paso 4 o con el 10 % de la poblacion."""
        previo = self.estado.datos["chequeos"].get("contratos")
        if previo and previo["ok"]:
            log.info(f"[completitud] Contratos ya verificado el {previo['fecha']} (sin cambios desde entonces): OK")
            return True
        self.revisar_tope()
        servidor = self.conteo_servidor(unidades)
        por_mes, ids, fuera = [], [], 0
        for u in unidades:
            df = pd.read_parquet(u["ruta"], columns=["id_contrato", "fecha_de_firma"])
            ids.append(df["id_contrato"])
            firma = pd.to_datetime(df["fecha_de_firma"], errors="coerce")
            fuera += int((~((firma >= u["ini"]) & (firma < u["fin"])) | ~df["id_contrato"].str.endswith(self.digito))
                         .sum())
            esperado = servidor.get(u["mes"], 0)
            tol = tolerancia(esperado, TOL_MES)
            por_mes.append({"mes": u["mes"], "anio": u["anio"], "servidor": esperado, "descargado": len(df),
                            "diferencia": len(df) - esperado, "tolerancia": tol, "ok": abs(len(df) - esperado) <= tol})
        duplicados = int(pd.concat(ids, ignore_index=True).duplicated().sum())
        por_anio = []
        for anio in sorted({m["anio"] for m in por_mes}):
            del_anio = [m for m in por_mes if m["anio"] == anio]
            n, srv = sum(m["descargado"] for m in del_anio), sum(m["servidor"] for m in del_anio)
            fila = {"anio": anio, "digito": self.digito, "meses": len(del_anio), "descargado": n, "servidor": srv,
                    "diferencia_servidor": n - srv}
            paso4 = CONTEO_PASO4.get(self.digito, {}).get(anio)
            if len(del_anio) == 12 and paso4 is not None:
                tol = tolerancia(paso4, TOL_PASO4)
                fila.update(paso4=paso4, diferencia_paso4=n - paso4, tolerancia_paso4=tol,
                            ok_paso4=abs(n - paso4) <= tol)
            if len(del_anio) == 12 and anio in POBLACION_PASO4:
                desvio = n / (POBLACION_PASO4[anio] / 10) - 1
                fila.update(diez_por_ciento=POBLACION_PASO4[anio] / 10, desvio_rel=round(desvio, 5),
                            ok_poblacion=abs(desvio) <= TOL_POBLACION)
            por_anio.append(fila)
        ok = (all(m["ok"] for m in por_mes) and duplicados == 0 and fuera == 0
              and all(f.get("ok_paso4", True) and f.get("ok_poblacion", True) for f in por_anio))
        resultado = {"fecha": ahora(), "ok": ok, "duplicados_id_contrato": duplicados, "filas_fuera_de_unidad": fuera,
                     "tolerancias": {"mes": TOL_MES, "paso4": TOL_PASO4, "poblacion": TOL_POBLACION},
                     "por_anio_digito": por_anio, "por_mes": por_mes}
        self.estado.datos["chequeos"]["contratos"] = resultado
        self.estado.guardar()
        malos = [m for m in por_mes if not m["ok"]]
        log.info(f"[completitud] Contratos: {sum(m['descargado'] for m in por_mes)} filas descargadas frente a "
                 f"{sum(m['servidor'] for m in por_mes)} del count(*) del servidor en {len(por_mes)} meses; "
                 f"meses fuera de tolerancia {len(malos)}; id_contrato repetidos {duplicados}; filas fuera de su "
                 f"unidad {fuera}")
        for m in malos:
            log.error(f"[completitud] {m['mes']}: descargado {m['descargado']}, servidor {m['servidor']} "
                      f"(tolerancia {m['tolerancia']})")
        for f in por_anio:
            log.info(f"[completitud] {f['anio']} dig {f['digito']}: descargado {f['descargado']}, servidor "
                     f"{f['servidor']}, paso 4 {f.get('paso4', '-')} (ok {f.get('ok_paso4', '-')}), 10 % poblacion "
                     f"{f.get('diez_por_ciento', '-')} (desvio {f.get('desvio_rel', '-')}, "
                     f"ok {f.get('ok_poblacion', '-')})")
        if not ok:
            log.error("[completitud] Contratos NO cuadra: no se sigue con Procesos ni Modificaciones")
        return ok

    # ---------------- modo de Contratos local (base B) ----------------
    def plan_contratos_local(self):
        """Registra en planes['contratos'] los archivos de la carpeta local usados y los descartados (fuera de rango)."""
        if not self.local.is_dir():
            raise SystemExit(f"--contratos-local: no existe la carpeta {self.local}")
        ini, fin = pd.Period(self.desde, "M"), pd.Period(self.hasta, "M")
        usados, descartados = 0, {}
        for ruta in sorted(self.local.glob("*.parquet")):
            try:
                periodo = pd.Period(ruta.stem, "M")
            except ValueError:
                log.warning(f"[contratos] {ruta.name}: el nombre no es AAAA-MM; se ignora")
                continue
            if ini <= periodo <= fin:
                usados += 1
            else:
                filas = len(pd.read_parquet(ruta, columns=["id_contrato"]))
                descartados[periodo.year] = descartados.get(periodo.year, 0) + filas
        plan = {"fuente": self.estado.datos["parametros"]["contratos_local"], "archivos_en_rango": usados,
                "filas_descartadas_por_anio": {str(a): n for a, n in sorted(descartados.items())},
                "filas_descartadas": sum(descartados.values())}
        self.estado.datos["planes"]["contratos"] = plan
        self.estado.guardar()
        log.info(f"[contratos] fuente local {plan['fuente']}: {usados} archivos en rango {self.desde} a {self.hasta}; "
                 f"descartados {plan['filas_descartadas']} filas fuera de rango {plan['filas_descartadas_por_anio']}")

    def copiar_unidad_local(self, unidad, i, n):
        """Lee el archivo del mes, quita las filas exactamente repetidas y guarda la unidad (mismo formato que F2)."""
        clave, ruta = unidad["clave"], unidad["ruta"]
        self.actual = ("contratos", clave)
        self.estado.marcar("contratos", clave, estado=PENDIENTE, inicio=ahora(), fin="", error="")
        t = time.perf_counter()
        try:
            df = pd.read_parquet(unidad["fuente"], columns=COLS["contratos"])
            leidas = len(df)
            df = df.drop_duplicates()   # filas repetidas por el servidor (2022-11, 2022-12 y 2024-07 tienen una)
            n_filas, n_bytes, sha = guardar_parquet(df.to_dict("records"), COLS["contratos"], ruta)
        except Exception as e:  # noqa: BLE001 - archivo ausente o sin columnas: la unidad queda fallida y se sigue
            self.estado.marcar("contratos", clave, estado=FALLIDO, fin=ahora(), error=f"{type(e).__name__}: {e}"[:300])
            log.error(f"[contratos] {i}/{n} {clave}: FALLIDO {type(e).__name__}: {str(e)[:200]}")
            return
        seg = time.perf_counter() - t
        nota = f"{leidas - n_filas} fila(s) repetida(s) del servidor eliminada(s)" if leidas != n_filas else ""
        self.estado.marcar("contratos", clave, estado=HECHO, filas=n_filas, paginas=0, bytes=n_bytes,
                           segundos=round(seg, 2), fin=ahora(), sha256=sha, error=nota)
        self.bajadas["contratos"] += 1
        if self.estado.datos["chequeos"].pop("contratos", None):
            self.estado.guardar()   # la unidad cambio: el chequeo se rehace
        log.info(f"[contratos] {i}/{n} {clave}: {n_filas} filas leidas de {unidad['fuente'].name}"
                 + (f" ({nota})" if nota else ""))

    def copiar_contratos_local(self, unidades):
        self.plan_contratos_local()
        por_pedir = self.estado.planificar("contratos", unidades)
        log.info(f"[contratos] {len(unidades)} unidades locales: {len(unidades) - len(por_pedir)} ya hechas, "
                 f"{len(por_pedir)} por escribir")
        for i, u in enumerate(por_pedir, 1):
            self.copiar_unidad_local(u, i, len(por_pedir))

    def chequear_contratos_local(self, unidades):
        """False si el total no es 167.997 (rango completo), hay id_contrato repetidos o filas fuera de su mes."""
        previo = self.estado.datos["chequeos"].get("contratos")
        if previo and previo["ok"]:
            log.info(f"[completitud] Contratos locales ya verificados el {previo['fecha']}: OK")
            return True
        por_mes, ids, fuera, fuente = [], [], 0, 0
        for u in unidades:
            df = pd.read_parquet(u["ruta"], columns=["id_contrato", "fecha_de_firma"])
            ids.append(df["id_contrato"])
            firma = pd.to_datetime(df["fecha_de_firma"], errors="coerce")
            fuera += int((~((firma >= u["ini"]) & (firma < u["fin"]))).sum())
            fuente += len(pd.read_parquet(u["fuente"], columns=["id_contrato"]))
            por_mes.append({"mes": u["mes"], "anio": u["anio"], "descargado": len(df)})
        duplicados = int(pd.concat(ids, ignore_index=True).duplicated().sum())
        total = sum(m["descargado"] for m in por_mes)
        completo = (self.desde, self.hasta) == (DESDE, HASTA)
        por_anio = [{"anio": a, "contratos": sum(m["descargado"] for m in por_mes if m["anio"] == a)}
                    for a in sorted({m["anio"] for m in por_mes})]
        ok = (duplicados == 0 and fuera == 0 and all(m["descargado"] > 0 for m in por_mes)
              and (not completo or total == CONTRATOS_LOCAL_ESPERADOS))
        resultado = {"fecha": ahora(), "ok": ok, "modo": "local", "total": total,
                     "esperado": CONTRATOS_LOCAL_ESPERADOS if completo else None,
                     "filas_fuente": fuente, "repetidas_eliminadas": fuente - total,
                     "duplicados_id_contrato": duplicados, "filas_fuera_de_unidad": fuera,
                     "por_anio": por_anio, "por_mes": por_mes}
        self.estado.datos["chequeos"]["contratos"] = resultado
        self.estado.guardar()
        log.info(f"[completitud] Contratos locales: {total} contratos (esperado "
                 f"{CONTRATOS_LOCAL_ESPERADOS if completo else '- (no es el rango completo)'}) en {len(unidades)} "
                 f"meses; {fuente - total} fila(s) repetida(s) eliminada(s); id_contrato repetidos {duplicados}; "
                 f"filas fuera de su mes {fuera}; por anio {[(f['anio'], f['contratos']) for f in por_anio]}")
        if not ok:
            log.error("[completitud] Contratos locales NO cuadra: no se sigue con Procesos ni Modificaciones")
        return ok

    def contar_lotes(self, dataset, lotes):
        """count(*) del servidor con el mismo $where de cada lote hecho, contra las filas descargadas. Se guarda por
        lote en _estado.json (reanudable); un lote cuyas filas cambiaron se vuelve a contar."""
        reg = self.estado.datos["chequeos"].setdefault("conteo_lotes", {}).setdefault(dataset, {})
        filas = {f["clave"]: int(f["filas"]) for f in self.estado.filas(dataset) if f["estado"] == HECHO}
        pendientes = [l for l in lotes if l["clave"] in filas
                      and not (l["clave"] in reg and reg[l["clave"]]["descargado"] == filas[l["clave"]])]
        log.info(f"[completitud] {dataset}: count(*) del servidor por lote: {len(lotes) - len(pendientes)} ya "
                 f"verificados, {len(pendientes)} por pedir")
        for i, l in enumerate(pendientes, 1):
            self.revisar_tope()
            clave = l["clave"]
            self.actual = (dataset, "conteo:" + clave)
            try:
                datos = self.agregado.consultar(config.DATASETS[dataset], {
                    "$select": "count(*) as n", "$where": self.where_lote(dataset, l["ids"])})
            except Exception as e:  # noqa: BLE001 - tras los reintentos: el lote queda sin verificar y se sigue
                self.volcar_peticiones(dataset, "conteo:" + clave)
                if isinstance(e, Throttled) or (isinstance(e, ValueError)
                                                and not isinstance(e, requests.exceptions.JSONDecodeError)):
                    raise
                log.error(f"[completitud] {dataset} {i}/{len(pendientes)} {clave}: conteo FALLIDO "
                          f"{type(e).__name__}: {str(e)[:200]}")
                continue
            self.volcar_peticiones(dataset, "conteo:" + clave)
            servidor = int(datos[0]["n"]) if datos else 0
            reg[clave] = {"servidor": servidor, "descargado": filas[clave], "ok": servidor == filas[clave]}
            self.estado.guardar()
            if servidor != filas[clave]:
                log.error(f"[completitud] {dataset} {clave}: descargadas {filas[clave]} filas, count(*) del servidor "
                          f"{servidor}")
            elif i % 25 == 0 or i == len(pendientes):
                log.info(f"[completitud] {dataset}: {i}/{len(pendientes)} conteos | transcurrido "
                         f"{self.hms(self.transcurrido())} | peticiones {self.peticiones()}")

    def estado_conteo(self):
        """Por conjunto: lotes, los que coinciden con el servidor, las diferencias y los que faltan por verificar."""
        reg = self.estado.datos["chequeos"].get("conteo_lotes", {})
        out = {}
        for dataset in ("procesos", "modificaciones"):
            coinciden, diferencias, sin_verificar = 0, [], 0
            filas = self.estado.filas(dataset)
            for f in filas:
                e = reg.get(dataset, {}).get(f["clave"])
                if f["estado"] != HECHO or e is None or e["descargado"] != int(f["filas"]):
                    sin_verificar += 1
                elif e["ok"]:
                    coinciden += 1
                else:
                    diferencias.append({"clave": f["clave"], "servidor": e["servidor"], "descargado": e["descargado"]})
            out[dataset] = {"lotes": len(filas), "coinciden": coinciden, "diferencias": diferencias,
                            "sin_verificar": sin_verificar}
        return out

    def resumen_lotes(self, dataset, lotes):
        # Lote por lote (los lotes no comparten IDs, asi que las cuentas se suman): no carga todo el conjunto en RAM
        columna = COLUMNA_FILTRO[dataset]
        hechos, filas, con_filas, duplicadas = 0, 0, 0, 0
        for l in lotes:
            if not l["ruta"].exists():
                continue
            df = pd.read_parquet(l["ruta"])
            hechos, filas = hechos + 1, filas + len(df)
            con_filas += int(df[columna].nunique())
            duplicadas += int(df.duplicated().sum())
        res = {"fecha": ahora(), "lotes": len(lotes), "hechos": hechos, "filas": filas,
               "ids_pedidos": sum(len(l["ids"]) for l in lotes), "ids_con_filas": con_filas,
               "filas_duplicadas_exactas": duplicadas}
        self.estado.datos["chequeos"][dataset] = res
        self.estado.guardar()
        log.info(f"[completitud] {dataset}: {res['hechos']}/{res['lotes']} lotes hechos, {res['filas']} filas, "
                 f"{res['ids_con_filas']}/{res['ids_pedidos']} IDs con al menos una fila, "
                 f"{res['filas_duplicadas_exactas']} filas duplicadas exactas")

    def escribir_completitud(self, codigo):
        """completitud.json: lo que F3 lee para saber si puede arrancar. ok solo si todo esta hecho y cuadra."""
        planes = self.estado.datos["planes"]
        chequeos = self.estado.datos["chequeos"]
        unidades, faltantes = {}, {}
        for ds in CONJUNTOS:
            filas = self.estado.filas(ds)
            cuenta = Counter(f["estado"] for f in filas)
            unidades[ds] = {"planificadas": len(filas), "hecho": cuenta[HECHO], "pendiente": cuenta[PENDIENTE],
                            "fallido": cuenta[FALLIDO]}
            faltantes[ds] = [{"clave": f["clave"], "estado": f["estado"], "error": f["error"]}
                             for f in filas if f["estado"] != HECHO]
        contratos_ok = bool(chequeos.get("contratos", {}).get("ok"))
        ok = (contratos_ok and all(ds in planes for ds in ("procesos", "modificaciones"))
              and all(unidades[ds]["planificadas"] > 0 and not faltantes[ds] for ds in CONJUNTOS))
        cuenta = self.estado_conteo() if self.local else None
        if cuenta:   # modo local: ademas, cada lote coincide con el count(*) del servidor
            ok = ok and all(c["lotes"] > 0 and c["coinciden"] == c["lotes"] for c in cuenta.values())
        codigo = codigo if (codigo != 0 or ok) else 2
        contenido = {
            "ok": ok, "codigo_salida": codigo, "motivo": "completo" if ok else (self.motivo or "pendientes"),
            "hora_limite": self.hora_limite.isoformat(timespec="seconds") if self.hora_limite else None,
            "fecha": ahora(), "corrida": self.corrida, "corte": self.estado.corte,
            "digito": self.digito, "desde": self.desde, "hasta": self.hasta, "destino": str(self.destino),
            "contratos_ok": contratos_ok,
            "conteos_por_anio_digito": chequeos.get("contratos", {}).get("por_anio_digito", []),
            "unidades": unidades, "faltantes": faltantes, "planes": planes, "chequeos": chequeos,
            "peticiones_corrida": self.peticiones(), "segundos_corrida": round(self.transcurrido(), 1)}
        if cuenta:
            contenido.update(modo="contratos_local", contratos_local=self.estado.datos["parametros"]["contratos_local"],
                             conteo_lotes=cuenta)
        escribir_atomico(self.ruta_completitud, json.dumps(contenido, indent=2, ensure_ascii=True, default=str))
        return ok, codigo

    # ---------------- corrida ----------------
    def correr(self, conjuntos):
        origen = f"contratos locales de {self.local}" if self.local else f"digito {self.digito}"
        log.info(f"corrida {self.corrida}: {origen}, meses {self.desde} a {self.hasta}, corte "
                 f"{self.estado.corte}, hora limite {self.hora_limite or '-'}, conjuntos {','.join(conjuntos)}, "
                 f"destino {self.destino}")
        unidades = self.unidades_contratos()
        if "contratos" in conjuntos:
            if self.local:
                self.copiar_contratos_local(unidades)
            else:
                self.recorrer("contratos", unidades, LIMITE_CONTRATOS, lambda u: u["where"])
        else:
            self.estado.planificar("contratos", unidades)
        if self.no_hechas("contratos"):
            log.error(f"[contratos] faltan {len(self.no_hechas('contratos'))} de {len(unidades)} unidades: Procesos "
                      "y Modificaciones esperan a que Contratos este completo")
            self.motivo = "contratos_pendientes"
            return 2
        if not (self.chequear_contratos_local(unidades) if self.local else self.chequear_contratos(unidades)):
            self.motivo = "chequeo_contratos_fallo"
            return 1
        por_conjunto = {}
        for dataset in ("procesos", "modificaciones"):
            lotes = por_conjunto[dataset] = self.lotes(dataset)
            if dataset in conjuntos:
                self.recorrer(dataset, lotes, LIMITE_LOTES, lambda u, d=dataset: self.where_lote(d, u["ids"]))
            else:
                n = len(self.estado.planificar(dataset, lotes))
                log.info(f"[{dataset}] no esta en --conjuntos: {n} lotes quedan pendientes")
            self.resumen_lotes(dataset, lotes)
        if self.local:   # count(*) del servidor por lote, al terminar las descargas (si falta tiempo, lo ultimo en caer)
            for dataset, lotes in por_conjunto.items():
                if dataset in conjuntos and not self.no_hechas(dataset):
                    self.contar_lotes(dataset, lotes)
            cuenta = self.estado_conteo()
            if any(c["diferencias"] for c in cuenta.values()):
                self.motivo = "conteo_lotes_no_cuadra"
                return 1
            if any(c["sin_verificar"] for c in cuenta.values()) and not any(self.no_hechas(ds) for ds in CONJUNTOS):
                self.motivo = "conteo_lotes_pendientes"
                return 2
        return 2 if any(self.no_hechas(ds) for ds in CONJUNTOS) else 0


def configurar_log(destino):
    destino.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    formato = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    consola = logging.StreamHandler(sys.stdout)
    consola.setLevel(logging.INFO)
    archivo = logging.FileHandler(destino / "_descarga.log", encoding="utf-8")
    archivo.setLevel(logging.DEBUG)
    for h in (consola, archivo):
        h.setFormatter(formato)
        log.addHandler(h)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Descarga F2 de SECOP II (reanudable).")
    ap.add_argument("--mes", help="limita la descarga a un mes de firma, AAAA-MM (prueba)")
    ap.add_argument("--digito", choices=list("0123456789"), help=f"ultimo digito de id_contrato (por defecto {DIGITO})")
    ap.add_argument("--destino", help="carpeta de salida (por defecto data/raw/f2/; con --contratos-local, "
                                      "data/raw/f2_b/)")
    ap.add_argument("--corte", help="fecha de corte AAAA-MM-DDTHH:MM:SS (por defecto, la hora de arranque; con "
                                    f"--contratos-local, {CORTE_LOCAL})")
    ap.add_argument("--contratos-local", nargs="?", const=str(CONTRATOS_LOCAL), metavar="CARPETA",
                    help="base B: lee Contratos de los archivos AAAA-MM.parquet de la exploracion (por defecto "
                         "00_exploracion/data/raw/contratos_chunks/), sin filtro por digito, y descarga solo Procesos "
                         "y Modificaciones")
    ap.add_argument("--conjuntos", default=",".join(CONJUNTOS),
                    help="subconjunto de contratos,procesos,modificaciones (por defecto, los tres)")
    ap.add_argument("--hasta", dest="hora_limite", metavar="AAAA-MM-DDTHH:MM",
                    help="hora limite local: al llegar se detiene limpiamente (codigo 4); reanudar con el mismo comando")
    args = ap.parse_args(argv)
    if args.contratos_local:
        if args.digito:
            ap.error("--contratos-local no filtra por digito: no use --digito")
        if not Path(args.contratos_local).is_dir():
            ap.error(f"--contratos-local: no existe la carpeta {args.contratos_local}")
        args.corte = args.corte or CORTE_LOCAL
        args.destino = args.destino or str(DESTINO_LOCAL)
    else:
        args.digito = args.digito or DIGITO
        args.destino = args.destino or str(DESTINO)
    if args.mes:
        pd.Period(args.mes, "M")   # valida el formato
    if args.corte:
        args.corte = datetime.fromisoformat(args.corte).strftime("%Y-%m-%dT%H:%M:%S")
    if args.hora_limite:
        limite = datetime.fromisoformat(args.hora_limite)
        if limite <= datetime.now():
            ap.error(f"--hasta {args.hora_limite} ya paso")
        args.hora_limite = limite.isoformat(timespec="seconds")
    conjuntos = [c.strip() for c in args.conjuntos.split(",") if c.strip()]
    if not conjuntos or set(conjuntos) - set(CONJUNTOS):
        ap.error(f"--conjuntos debe ser un subconjunto de {','.join(CONJUNTOS)}")
    configurar_log(Path(args.destino).resolve())
    time.sleep(1.0)   # el limitador es por proceso: separa de la ultima peticion de una corrida anterior
    d = Descarga(args)
    codigo = 1
    try:
        codigo = d.correr(conjuntos)
    except TopeTiempo:
        log.warning(f"hora limite {args.hora_limite} alcanzada: se detiene limpiamente (ninguna unidad a medias); "
                    "reanudar con el mismo comando")
        d.motivo, codigo = "hora_limite", 4
    except Throttled as e:
        log.error(f"el portal limito el uso, se detiene todo: {e}")
        d.motivo, codigo = "throttled", 3
    except ValueError as e:
        log.error(f"consulta invalida, se detiene: {e}")
        d.motivo, codigo = "consulta_invalida", 1
    except KeyboardInterrupt:
        log.warning("interrumpido: el estado en disco queda consistente; reanudar con el mismo comando")
        d.motivo, codigo = "interrumpido", 130
    except Exception as e:  # noqa: BLE001 - p. ej. el count(*) de control fallo tras los reintentos
        log.error(f"fallo {type(e).__name__}: {str(e)[:300]}; el estado en disco queda consistente")
        d.motivo, codigo = "error", 1
    finally:
        d.volcar_peticiones(*d.actual)
        ok, codigo = d.escribir_completitud(codigo)
        estados = {ds: dict(Counter(f["estado"] for f in d.estado.filas(ds))) for ds in CONJUNTOS}
        log.info(f"fin de la corrida {d.corrida}: codigo {codigo}, completitud ok={ok}, peticiones {d.peticiones()}, "
                 f"{d.hms(d.transcurrido())}, unidades bajadas {d.bajadas}, manifiesto {estados}, "
                 f"completitud en {d.ruta_completitud}")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
