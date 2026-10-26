"""Cliente HTTP minimo para datos.gov.co (Socrata), con limitador de tasa start-to-start y App Token.

Reglas que aplica (robots.txt de datos.gov.co con Crawl-delay: 1, acordado en la F1):
- Maximo 1 peticion por segundo, en secuencia y sin paralelismo. El segundo se mide entre el INICIO de una peticion
  y el INICIO de la siguiente: si una peticion tarda mas de 1 s, la siguiente sale de inmediato.
- Solo rutas permitidas: /resource/<id>.json (SODA 2.x), /api/v3/views/<id>/query.json y /api/views/<id>(.json)
  (metadatos). Nunca /OData.svc/, /api/odata/, /api/collocate* ni /views/INLINE/rows.json (method=clustered2).
- El App Token (SOCRATA_APP_TOKEN) se lee en tiempo de ejecucion del entorno o de final-project/.env y solo viaja en
  el header X-App-Token. Nunca se imprime, no se loguea y no va en la URL.
- Timeout de 30 s y hasta 3 intentos con espera creciente; ante 403/429 se detiene todo (Throttled), sin reintentar.
- Con reintento_exponencial=True (lo usa src/descarga.py): espera exponencial entre intentos (espera_reintento x 2^k),
  tambien reintenta 429 y respeta Retry-After en 429 y 5xx. 403, un 429 que sigue en el ultimo intento o un
  Retry-After mayor que espera_max detienen todo (Throttled). Cada reintento vuelve a pasar por el limitador.

Todas las instancias comparten por defecto el mismo limitador y el mismo candado, asi que dentro de un proceso las
peticiones a datos.gov.co salen de a una aunque haya varios clientes. Lo usan los scripts de la F1 y src/descarga.py.
"""
import email.utils
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import config

NOMBRE_TOKEN = "SOCRATA_APP_TOKEN"
_RUTAS_PERMITIDAS = re.compile(
    r"^/(resource/[a-z0-9]{4}-[a-z0-9]{4}\.json"
    r"|api/v3/views/[a-z0-9]{4}-[a-z0-9]{4}/query\.json"
    r"|api/views/[a-z0-9]{4}-[a-z0-9]{4}(\.json)?)$"
)


class Throttled(RuntimeError):
    """El portal limito el uso (403/429). Se detiene todo: no se reintenta."""


class LimitadorTasa:
    """Garantiza al menos `intervalo` segundos entre el inicio de una peticion y el inicio de la siguiente.

    No es una espera fija despues de cada respuesta: si la peticion anterior tardo mas que `intervalo`, la siguiente
    sale de inmediato. `reloj` y `dormir` se pueden reemplazar para probarlo sin red y sin esperar.
    """

    def __init__(self, intervalo=1.0, reloj=time.perf_counter, dormir=time.sleep):
        self.intervalo = float(intervalo)
        self.reloj = reloj
        self._dormir = dormir
        self._ultimo_inicio = None
        self._candado = threading.Lock()

    def esperar(self):
        """Bloquea hasta que se puede iniciar la siguiente peticion; registra y devuelve ese instante de inicio."""
        with self._candado:
            ahora = self.reloj()
            if self._ultimo_inicio is not None:
                objetivo = self._ultimo_inicio + self.intervalo
                while ahora < objetivo:          # en bucle: time.sleep puede despertar un poco antes
                    self._dormir(objetivo - ahora)
                    ahora = self.reloj()
            self._ultimo_inicio = ahora
            return ahora


LIMITADOR = LimitadorTasa(1.0)      # compartido por todo el proceso
_EN_CURSO = threading.Lock()        # una sola peticion a la vez, aunque haya hilos


def leer_token(ruta_env=None, nombre=NOMBRE_TOKEN):
    """Devuelve el App Token: variable de entorno `nombre` o linea `nombre=valor` de final-project/.env.

    Lector simple de lineas (sin python-dotenv). Nunca imprime el valor; si falta, el error solo dice donde se busco.
    """
    valor = os.environ.get(nombre, "").strip()
    ruta = Path(ruta_env) if ruta_env else config.RAIZ / ".env"
    if not valor and ruta.exists():
        for linea in ruta.read_text(encoding="utf-8-sig").splitlines():
            clave, sep, resto = linea.strip().partition("=")
            clave = clave.removeprefix("export ").strip()
            if sep and clave == nombre:
                resto = resto.strip()
                if resto[:1] in ("'", '"') and resto[-1:] == resto[:1] and len(resto) >= 2:
                    resto = resto[1:-1]
                else:
                    resto = resto.split(" #")[0].strip()
                valor = resto
                break
    if not valor:
        raise RuntimeError(f"No se encontro {nombre} en el entorno ni en {ruta}")
    return valor


def segundos_retry_after(valor):
    """Retry-After en segundos (entero o fecha HTTP); None si no llega o no se entiende."""
    if not valor:
        return None
    valor = str(valor).strip()
    try:
        return max(0.0, float(valor))
    except ValueError:
        pass
    try:
        fecha = email.utils.parsedate_to_datetime(valor)
    except (TypeError, ValueError):
        return None
    if fecha.tzinfo is None:
        fecha = fecha.replace(tzinfo=timezone.utc)
    return max(0.0, (fecha - datetime.now(timezone.utc)).total_seconds())


class ClienteSocrata:
    """GET a datos.gov.co con el limitador, el App Token en X-App-Token, timeout, reintentos y registro de tiempos."""

    def __init__(self, token=None, limitador=None, timeout=30, intentos=3, espera_reintento=5.0, sesion=None,
                 base_url=config.BASE_URL, dormir=time.sleep, reintento_exponencial=False, espera_max=300.0):
        self._token = token if token is not None else leer_token()
        self.limitador = limitador or LIMITADOR
        self.timeout = timeout
        self.intentos = intentos
        self.espera_reintento = espera_reintento
        self.reintento_exponencial = reintento_exponencial
        self.espera_max = espera_max
        self._sesion = sesion or requests.Session()
        self.base_url = base_url.rstrip("/")
        self._dormir = dormir
        self.registro = []   # una entrada por peticion enviada (incluye reintentos): inicio, segundos, status

    def __repr__(self):
        return f"ClienteSocrata(base_url={self.base_url!r}, token=***, peticiones={self.peticiones})"

    @property
    def peticiones(self):
        return len(self.registro)

    def _ocultar(self, texto):
        return str(texto).replace(self._token, "***") if self._token else str(texto)

    def get(self, ruta, params=None):
        """GET a una ruta permitida; devuelve el JSON de la respuesta."""
        if not _RUTAS_PERMITIDAS.match(ruta):
            raise ValueError(f"ruta no permitida para datos.gov.co: {ruta}")
        url = self.base_url + ruta
        cabeceras = {"X-App-Token": self._token, "Accept": "application/json"}
        motivo = ""
        for intento in range(1, self.intentos + 1):
            with _EN_CURSO:
                inicio = self.limitador.esperar()
                status = None
                try:
                    r = self._sesion.get(url, params=params, headers=cabeceras, timeout=self.timeout)
                    status = r.status_code
                except requests.RequestException as e:
                    r, motivo = None, type(e).__name__
                finally:
                    self.registro.append({"inicio": inicio, "segundos": self.limitador.reloj() - inicio,
                                          "status": status, "ruta": ruta, "intento": intento})
            if self.reintento_exponencial:
                espera = self.espera_reintento * 2 ** (intento - 1)
            else:
                espera = self.espera_reintento * intento
            if r is not None:
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 403 or (r.status_code == 429 and not self.reintento_exponencial):
                    raise Throttled(f"{r.status_code} en {ruta}: {self._ocultar(r.text[:300])}")
                if r.status_code == 400:   # error de la consulta: reintentar no sirve
                    raise ValueError(f"400 en {ruta}: {self._ocultar(r.text[:300])}")
                motivo = f"status {r.status_code}"
                if self.reintento_exponencial and (r.status_code == 429 or r.status_code >= 500):
                    pedida = segundos_retry_after(r.headers.get("Retry-After"))
                    if pedida is not None and pedida > self.espera_max:
                        raise Throttled(f"{r.status_code} en {ruta}: Retry-After de {pedida:.0f} s "
                                        f"(mas que {self.espera_max:.0f} s)")
                    if pedida is not None:
                        espera = max(espera, pedida)
                    if r.status_code == 429 and intento == self.intentos:
                        raise Throttled(f"429 en {ruta} tras {self.intentos} intentos: "
                                        f"{self._ocultar(r.text[:300])}")
            if intento < self.intentos:
                self._dormir(min(espera, self.espera_max) if self.reintento_exponencial else espera)
        raise RuntimeError(f"Fallo {ruta} tras {self.intentos} intentos ({motivo})")

    def consultar(self, dataset_id, params):
        """Consulta SODA 2.x: /resource/<id>.json con parametros SoQL ($select, $where, $order, $limit, $offset)."""
        return self.get(f"/resource/{dataset_id}.json", params)
