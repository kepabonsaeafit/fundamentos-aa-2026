"""Limitador start-to-start y cliente de datos.gov.co, sin red: reloj simulado y sesion falsa.

Peticiones falsas de 0,2 s y 1,5 s: tras una de 0,2 s la siguiente espera hasta cumplir 1 s desde el inicio de la
anterior; tras una de 1,5 s sale de inmediato. Ejecutar desde final-project/:  pytest
"""
import pytest

from src.cliente_socrata import ClienteSocrata, LimitadorTasa, Throttled, leer_token

TOKEN = "token-de-prueba-123"


class Reloj:
    """Reloj simulado: dormir() y las peticiones falsas avanzan el tiempo sin esperar de verdad."""

    def __init__(self):
        self.t = 0.0
        self.dormido = 0.0

    def __call__(self):
        return self.t

    def dormir(self, s):
        self.t += s
        self.dormido += s


class Respuesta:
    def __init__(self, status, cuerpo, headers=None):
        self.status_code = status
        self._cuerpo = cuerpo
        self.text = str(cuerpo)
        self.headers = headers or {}

    def json(self):
        return self._cuerpo


class SesionFalsa:
    """Cada get tarda lo indicado en `duraciones` (en el reloj simulado) y guarda lo que se envio.

    `status` y `cabeceras` pueden ser listas: una entrada por llamada."""

    def __init__(self, reloj, duraciones, status=200, cuerpo=None, cabeceras=None):
        self.reloj, self.duraciones, self.status, self.cuerpo = reloj, list(duraciones), status, cuerpo
        self.cabeceras = cabeceras
        self.llamadas = []

    def get(self, url, params=None, headers=None, timeout=None):
        k = len(self.llamadas)
        self.llamadas.append({"inicio": self.reloj.t, "url": url, "params": params, "headers": headers})
        self.reloj.t += self.duraciones.pop(0)
        status = self.status[k] if isinstance(self.status, list) else self.status
        cab = self.cabeceras[k] if isinstance(self.cabeceras, list) else self.cabeceras
        return Respuesta(status, self.cuerpo if self.cuerpo is not None else [{"ok": "1"}], cab)


def _cliente(duraciones, reintento_exponencial=False, **kw):
    reloj = Reloj()
    limitador = LimitadorTasa(1.0, reloj=reloj, dormir=reloj.dormir)
    sesion = SesionFalsa(reloj, duraciones, **kw)
    return ClienteSocrata(token=TOKEN, limitador=limitador, sesion=sesion, dormir=reloj.dormir,
                          reintento_exponencial=reintento_exponencial), sesion, reloj


def test_limitador_mide_de_inicio_a_inicio():
    cliente, sesion, reloj = _cliente([0.2, 1.5, 0.2, 0.2])
    for _ in range(4):
        cliente.consultar("u8cx-r425", {"$limit": 1})
    inicios = [c["inicio"] for c in sesion.llamadas]
    # 0,2 s -> espera 0,8 s; 1,5 s -> sale de inmediato (2,5); 0,2 s -> espera hasta 3,5
    assert inicios == pytest.approx([0.0, 1.0, 2.5, 3.5])
    assert reloj.dormido == pytest.approx(0.8 + 0.8)
    assert cliente.peticiones == 4


def test_token_solo_en_el_header():
    cliente, sesion, _ = _cliente([0.2])
    cliente.consultar("u8cx-r425", {"$where": "id_contrato in ('CO1.PCCNTR.1')"})
    llamada = sesion.llamadas[0]
    assert llamada["headers"]["X-App-Token"] == TOKEN
    assert TOKEN not in llamada["url"] and TOKEN not in str(llamada["params"])
    assert TOKEN not in repr(cliente)


def test_rutas_prohibidas_no_salen():
    cliente, sesion, _ = _cliente([])
    for ruta in ["/OData.svc/jbjy-vk9h", "/api/odata/v4/jbjy-vk9h", "/api/collocate", "/views/INLINE/rows.json"]:
        with pytest.raises(ValueError):
            cliente.get(ruta)
    assert sesion.llamadas == []


def test_429_detiene_sin_reintentar_y_oculta_el_token():
    cliente, sesion, _ = _cliente([0.1], status=429, cuerpo=f"limite excedido para {TOKEN}")
    with pytest.raises(Throttled) as e:
        cliente.consultar("u8cx-r425", {})
    assert TOKEN not in str(e.value)
    assert len(sesion.llamadas) == 1


def test_reintentos_tambien_pasan_por_el_limitador():
    cliente, sesion, _ = _cliente([0.1, 0.1, 0.1], status=500)
    with pytest.raises(RuntimeError):
        cliente.consultar("u8cx-r425", {})
    inicios = [c["inicio"] for c in sesion.llamadas]
    assert len(inicios) == 3 and all(b - a >= 1.0 for a, b in zip(inicios, inicios[1:]))


def test_exponencial_reintenta_429_respetando_retry_after():
    cliente, sesion, _ = _cliente([0.1, 0.1], status=[429, 200], cabeceras=[{"Retry-After": "7"}, {}],
                                  reintento_exponencial=True)
    assert cliente.consultar("u8cx-r425", {}) == [{"ok": "1"}]
    inicios = [c["inicio"] for c in sesion.llamadas]
    assert inicios[1] - inicios[0] == pytest.approx(0.1 + 7.0)   # Retry-After (7 s) > backoff (5 s)


def test_exponencial_5xx_espera_creciente_y_429_final_detiene():
    cliente, sesion, _ = _cliente([0.1, 0.1, 0.1], status=500, reintento_exponencial=True)
    with pytest.raises(RuntimeError):
        cliente.consultar("u8cx-r425", {})
    inicios = [c["inicio"] for c in sesion.llamadas]
    assert [round(b - a, 6) for a, b in zip(inicios, inicios[1:])] == [5.1, 10.1]   # 5 s y 10 s tras cada respuesta
    cliente, sesion, _ = _cliente([0.1] * 3, status=429, cuerpo=f"limite {TOKEN}", reintento_exponencial=True)
    with pytest.raises(Throttled) as e:
        cliente.consultar("u8cx-r425", {})
    assert len(sesion.llamadas) == 3 and TOKEN not in str(e.value)
    cliente, sesion, _ = _cliente([0.1], status=429, cabeceras={"Retry-After": "3600"}, reintento_exponencial=True)
    with pytest.raises(Throttled):
        cliente.consultar("u8cx-r425", {})
    assert len(sesion.llamadas) == 1   # Retry-After mayor que espera_max: se detiene sin esperar


def test_leer_token_de_env(tmp_path, monkeypatch):
    monkeypatch.delenv("SOCRATA_APP_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text("# comentario\nOTRA=1\nSOCRATA_APP_TOKEN=\"abc123\"\n", encoding="utf-8")
    assert leer_token(env) == "abc123"
    with pytest.raises(RuntimeError) as e:
        leer_token(tmp_path / "no_existe.env")
    assert "abc123" not in str(e.value)
