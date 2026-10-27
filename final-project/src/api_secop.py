"""Consulta de un contrato y de su proceso en datos.gov.co para la app (G1).

Lo que hace:
- Pide a la API SOLO las columnas de entrada de construir_features (mas el nombre de la entidad, que solo se muestra).
  Las columnas de fuga (estado, fecha de fin, valor, pagos, liquidacion...) y las de datos personales no se solicitan,
  asi que no llegan aunque el portal las tenga actualizadas despues de la firma.
- Usa el cliente de src/cliente_socrata.py: 1 peticion por segundo, solo rutas permitidas, timeout de 30 s y 3 intentos.
- Arma las features con las mismas funciones del entrenamiento: elegir_proceso -> unir_contrato_proceso -> construir_features.
  La API devuelve las fechas con hora; construir_features las trunca al dia, igual que en la base de entrenamiento.
- Si la API falla y hay un respaldo local (filas de la base), responde desde ahi y lo avisa (fuente = "respaldo").

El App Token es opcional: sin token la API responde igual, con menos cupo. Se lee de SOCRATA_APP_TOKEN (entorno o
final-project/.env) o se pasa como argumento (la app lo toma de st.secrets). Nunca se imprime ni se guarda.
"""
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

from . import config
from . import features as F
from . import uniones
from .cliente_socrata import NOMBRE_TOKEN, ClienteSocrata, Throttled, leer_token

# El id de un contrato de SECOP II es CO1.PCCNTR.<numero>; sale tal cual de la pagina del contrato o dentro de su enlace.
PATRON_ID_CONTRATO = re.compile(r"CO1\.PCCNTR\.\d+", re.IGNORECASE)
# Otros numeros de SECOP II que el usuario puede confundir con el del contrato: el del proceso (CO1.BDOS.n, CO1.REQ.n...)
PATRON_ID_PROCESO = re.compile(r"CO1\.(?:BDOS|REQ|NTC|PPI)\.\d+", re.IGNORECASE)
_ID_SEGURO = re.compile(r"^[A-Za-z0-9._-]{1,60}$")      # lo unico que se deja pasar a un $where

# Columnas pedidas a la API. Salen de las listas verificadas del proyecto; los assert impiden que entre fuga o PII.
COLS_CONTRATO_API = ["id_contrato", "proceso_de_compra", "nombre_entidad"] + F.COLS_CONTRATO
COLS_PROCESO_API = (["id_del_portafolio", "id_del_proceso", "adjudicado", "fecha_adjudicacion", "valor_total_adjudicacion"]
                    + [c.removeprefix("p_") for c in F.COLS_PROCESO])
assert not set(COLS_CONTRATO_API) & F.COLUMNAS_FUGA
assert not set(COLS_PROCESO_API) & F.COLUMNAS_FUGA
assert not (set(COLS_CONTRATO_API) | set(COLS_PROCESO_API)) & config.PII

# Columnas que se leen del respaldo local (esquema de la base v1)
COLS_RESPALDO = ["id_contrato", "proceso_de_compra", "nombre_entidad", "eval_en_procesos"] + F.ENTRADAS

LOTE_RESPALDO = 20_000           # filas por lote al buscar un contrato en el parquet de respaldo
ANIO_MIN, ANIO_MAX = config.ANIO_INICIO, config.ANIO_DERIVA     # periodo con el que se entreno y evaluo el modelo
UMBRAL_FALTANTES = 0.40          # mas de esta fraccion de features sin dato -> aviso
TTL_CACHE = 3600                 # segundos que se recuerda una consulta (para no repetir peticiones)

# Claves de aviso que devuelve la consulta; la app las traduce a texto.
AVISO_SIN_PROCESO = "sin_proceso"
AVISO_FUERA_DE_PERIODO = "fuera_de_periodo"
AVISO_FECHA_INVALIDA = "fecha_firma_invalida"
AVISO_MUCHOS_FALTANTES = "muchos_faltantes"
AVISO_DUPLICADO = "contrato_duplicado"
AVISO_ENTIDAD_SIN_HISTORIAL = "entidad_sin_historial"


class ContratoNoEncontrado(LookupError):
    """El id no existe en Contratos (o el portal no lo devolvio)."""


class IdInvalido(ValueError):
    """El texto no contiene un id de contrato de SECOP II."""


class ApiNoDisponible(RuntimeError):
    """No se pudo consultar datos.gov.co (red, limite de uso o error del servidor) y no hay respaldo local."""


@dataclass
class Consulta:
    id_contrato: str
    features: pd.DataFrame                  # una fila, exactamente F.FEATURES
    contrato: dict                          # solo para mostrar al usuario; no entra al modelo
    avisos: list = field(default_factory=list)
    fuente: str = "api"                     # "api" o "respaldo"


class _SesionSinCabecerasVacias(requests.Session):
    """Sin token no se manda X-App-Token: un header con valor vacio puede hacer que el portal rechace la peticion."""

    def request(self, method, url, **kwargs):
        cabeceras = kwargs.get("headers") or {}
        kwargs["headers"] = {k: v for k, v in cabeceras.items() if v not in ("", None)}
        return super().request(method, url, **kwargs)


def token_de(secretos):
    """App Token de un mapeo tipo st.secrets, o None si no hay secretos o no esta la clave (nunca falla)."""
    try:
        return (secretos.get(NOMBRE_TOKEN) or None) if secretos is not None else None
    except Exception:       # st.secrets lanza si no existe secrets.toml
        return None


def crear_cliente(token=None):
    """Token del argumento, o de SOCRATA_APP_TOKEN / final-project/.env, o ninguno."""
    if token is None:
        try:
            token = leer_token()
        except RuntimeError:
            token = ""
    return ClienteSocrata(token=token, sesion=_SesionSinCabecerasVacias())


def extraer_id_contrato(texto):
    """Devuelve el id (CO1.PCCNTR.n) que aparezca en el texto o en el enlace pegado; si no hay, lanza IdInvalido.

    El portal define id_contrato como el identificador del contrato firmado generado por la plataforma. La referencia
    del contrato la genera cada entidad (no es unica) y el numero del proceso identifica el proceso, no el contrato:
    ninguno de los dos sirve para esta consulta, y se le explica al usuario.
    """
    m = PATRON_ID_CONTRATO.search(texto or "")
    if m:
        return m.group(0).upper()
    if PATRON_ID_PROCESO.search(texto or ""):
        raise IdInvalido("Ese número es de un proceso de compra, no de un contrato. Busca en la página del contrato firmado "
                         "el número que empieza por CO1.PCCNTR (por ejemplo CO1.PCCNTR.1234567).")
    raise IdInvalido("No encontré un número de contrato de SECOP II. Tiene la forma CO1.PCCNTR.1234567 "
                     "y aparece en la página del contrato o dentro de su enlace.")


def _donde_igual(columna, valor):
    if not _ID_SEGURO.match(str(valor)):
        raise ValueError(f"valor no permitido en la consulta: {valor!r}")
    return f"{columna} = '{valor}'"


def consultar_contrato(cliente, id_contrato):
    """Filas de Contratos con el id dado, solo con COLS_CONTRATO_API (texto como lo devuelve la API)."""
    filas = cliente.consultar(config.DATASETS["contratos"], {
        "$select": ",".join(COLS_CONTRATO_API), "$where": _donde_igual("id_contrato", id_contrato), "$limit": 5})
    return pd.DataFrame(filas)


def consultar_procesos(cliente, portafolio):
    """Filas de Procesos del portafolio (una por lote o adjudicacion), solo con COLS_PROCESO_API."""
    filas = cliente.consultar(config.DATASETS["procesos"], {
        "$select": ",".join(COLS_PROCESO_API), "$where": _donde_igual("id_del_portafolio", portafolio), "$limit": 200})
    return pd.DataFrame(filas)


def _datos_para_mostrar(fila):
    """Campos legibles del contrato y del proceso (ya unidos). No son features: no se usan en el modelo."""
    def texto(c):
        v = fila.get(c)
        return None if v is None or pd.isna(v) else str(v)
    firma = pd.to_datetime(fila.get("fecha_de_firma"), errors="coerce", format="ISO8601")
    return {
        "id_contrato": texto("id_contrato"), "proceso_de_compra": texto("proceso_de_compra"),
        "nombre_entidad": texto("nombre_entidad"), "objeto": texto("objeto_del_contrato"),
        "fecha_de_firma": None if pd.isna(firma) else firma.date().isoformat(),
        "modalidad": texto("modalidad_de_contratacion"), "tipo_de_contrato": texto("tipo_de_contrato"),
        "departamento": texto("departamento"), "ciudad": texto("ciudad"),
        "precio_base": pd.to_numeric(fila.get("p_precio_base"), errors="coerce"),
    }


def _avisos(feats, contrato, hay_proceso, duplicado=False):
    avisos = []
    if feats["mes_firma"].isna().iloc[0]:
        avisos.append(AVISO_FECHA_INVALIDA)
    else:
        a = pd.to_datetime(contrato.get("fecha_de_firma"), errors="coerce")
        if pd.notna(a) and not (ANIO_MIN <= a.year <= ANIO_MAX):
            avisos.append(AVISO_FUERA_DE_PERIODO)
    if not hay_proceso:
        avisos.append(AVISO_SIN_PROCESO)
    if feats.iloc[0].isna().mean() > UMBRAL_FALTANTES:
        avisos.append(AVISO_MUCHOS_FALTANTES)
    if duplicado:
        avisos.append(AVISO_DUPLICADO)
    return avisos


def avisos_del_modelo(feats, meta):
    """Avisos que dependen del artefacto del modelo, no del contrato.

    Si el .json del modelo trae `entidades_vistas` (los NIT de las entidades con contratos en el entrenamiento), avisa
    cuando la entidad del contrato no esta ahi: el modelo no conoce sus habitos y solo se apoya en el contrato. Si el
    .json no trae la lista (el dummy no la trae), no se puede saber y no se avisa nada.
    """
    vistas = (meta or {}).get("entidades_vistas")
    if not vistas:
        return []
    nit = feats.iloc[0]["nit_entidad"]
    if pd.isna(nit) or str(nit) not in {str(v) for v in vistas}:
        return [AVISO_ENTIDAD_SIN_HISTORIAL]
    return []


def consulta_desde_filas(contratos, procesos):
    """Contratos (1 fila) + Procesos (las filas del portafolio, puede ser vacio) -> Consulta. Sin red."""
    duplicado = bool(contratos["id_contrato"].duplicated().any())
    contratos = contratos.drop_duplicates("id_contrato").reset_index(drop=True)
    hay_proceso = procesos is not None and len(procesos) > 0
    p = uniones.elegir_proceso(procesos) if hay_proceso else pd.DataFrame(columns=["id_del_portafolio"])
    unido = uniones.unir_contrato_proceso(contratos, p)
    feats = F.construir_features(unido)
    fila = unido.iloc[0].to_dict()
    contrato = _datos_para_mostrar(fila)
    return Consulta(id_contrato=str(contrato["id_contrato"]), features=feats.iloc[[0]], contrato=contrato,
                    avisos=_avisos(feats.iloc[[0]], contrato, hay_proceso, duplicado))


def _fila_de_respaldo(respaldo, id_contrato):
    """Fila del contrato en el respaldo, o None. `respaldo` es un DataFrame ya cargado o la ruta a un parquet con el
    esquema de la base v1; con una ruta se lee solo esa fila (la base completa ocupa ~400 MB en memoria)."""
    if respaldo is None:
        return None
    if isinstance(respaldo, (str, Path)):
        if not Path(respaldo).exists():
            return None
        # Por lotes: read_parquet(filters=...) solo descarta grupos de filas por estadisticas y, con ids sin ordenar,
        # lee casi todo el archivo (~1 GB de RAM). Asi el pico es de un lote (~20 MB).
        for lote in pq.ParquetFile(respaldo).iter_batches(batch_size=LOTE_RESPALDO, columns=COLS_RESPALDO):
            coincide = pc.equal(lote["id_contrato"], id_contrato)
            if pc.any(coincide).as_py():
                return lote.filter(coincide).to_pandas().iloc[[0]]
        return None
    fila = respaldo[respaldo["id_contrato"] == id_contrato]
    return fila.iloc[[0]] if len(fila) else None


def _consulta_de_fila(fila, id_contrato):
    feats = F.construir_features(fila)
    contrato = _datos_para_mostrar(fila.iloc[0].to_dict())
    hay_proceso = bool(fila.iloc[0].get("eval_en_procesos", True))
    return Consulta(id_contrato=id_contrato, features=feats, contrato=contrato,
                    avisos=_avisos(feats, contrato, hay_proceso), fuente="respaldo")


def consulta_desde_respaldo(respaldo, id_contrato):
    """Misma salida que la consulta a la API, pero desde la base local (DataFrame o ruta al parquet)."""
    fila = _fila_de_respaldo(respaldo, id_contrato)
    if fila is None:
        raise ContratoNoEncontrado(id_contrato)
    return _consulta_de_fila(fila, id_contrato)


_CACHE = {}


def obtener_consulta(texto, cliente=None, respaldo=None, ttl=TTL_CACHE):
    """Punto de entrada de la app: texto pegado (id o enlace) -> Consulta.

    Con `respaldo` (DataFrame o ruta a la base) responde desde ahi si la API no esta disponible, y marca fuente="respaldo".
    Lanza IdInvalido, ContratoNoEncontrado o ApiNoDisponible, con mensajes aptos para mostrar.
    """
    id_contrato = extraer_id_contrato(texto)
    guardado = _CACHE.get(id_contrato)
    if guardado and time.time() - guardado[0] < ttl:
        return guardado[1]
    cliente = cliente or crear_cliente()
    try:
        contratos = consultar_contrato(cliente, id_contrato)
        if contratos.empty:
            raise ContratoNoEncontrado(f"No encontré el contrato {id_contrato} en datos.gov.co.")
        portafolio = contratos.iloc[0].get("proceso_de_compra")
        procesos = consultar_procesos(cliente, portafolio) if isinstance(portafolio, str) and portafolio else None
        consulta = consulta_desde_filas(contratos, procesos)
    except (Throttled, RuntimeError, requests.RequestException, ValueError) as e:
        # ValueError: el cliente lo lanza cuando la API responde 400 (consulta rechazada). IdInvalido (otro ValueError)
        # sale antes del try, asi que no pasa por aqui.
        fila = _fila_de_respaldo(respaldo, id_contrato)
        if fila is not None:
            return _consulta_de_fila(fila, id_contrato)
        raise ApiNoDisponible("No pude consultar datos.gov.co en este momento. Intenta de nuevo en unos minutos.") from e
    _CACHE[id_contrato] = (time.time(), consulta)
    return consulta
