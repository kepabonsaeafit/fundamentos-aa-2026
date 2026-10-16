"""Codigo de descarga usado para armar data/raw/ (referencia; NO se ejecuta desde exploracion_secop.ipynb).

Se corrio una sola vez el 2026-10-14 contra datos.gov.co (Socrata, sin App Token). Queda aqui para poder
reproducir la muestra. Notas:
- Los lotes de IN (...) fueron de 150 IDs; los timeouts fueron de 300 s con hasta 4 intentos (la version
  posterior pedira 30 s y 3 intentos, con cache por lote y log de progreso; no se aplico porque no hubo
  mas descargas).
- Las constantes SEL_CONTRATOS / SEL_PROC estan abreviadas: SEL_CONTRATOS son todas las columnas de la
  metadata de Contratos menos las de datos personales (PII).
"""
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

BASE = "https://www.datos.gov.co"
SEED = 42
DS = {"contratos": "jbjy-vk9h", "procesos": "p6dx-8zbt", "ejecucion": "mfmm-jqmq",
      "adiciones": "cb9c-h8sn", "modificaciones": "u8cx-r425"}
CUOTA_MES = 2000
LOTE = 150

# Columnas con datos personales: nunca se descargan.
PII = {
    "nombre_representante_legal", "nacionalidad_representante_legal", "domicilio_representante_legal",
    "tipo_de_identificaci_n_representante_legal", "identificaci_n_representante_legal", "g_nero_representante_legal",
    "nombre_del_banco", "tipo_de_cuenta", "n_mero_de_cuenta",
    "nombre_ordenador_del_gasto", "tipo_de_documento_ordenador_del_gasto", "n_mero_de_documento_ordenador_del_gasto",
    "nombre_supervisor", "tipo_de_documento_supervisor", "n_mero_de_documento_supervisor",
    "nombre_ordenador_de_pago", "tipo_de_documento_ordenador_de_pago", "n_mero_de_documento_ordenador_de_pago",
    "documento_proveedor", "proveedor_adjudicado", "direcci_n_de_ejecuci_n_del_contrato",
}

SEL_PROC = ["id_del_proceso", "id_del_portafolio", "precio_base", "duracion", "unidad_de_duracion",
            "fecha_de_publicacion_del", "fecha_de_recepcion_de", "fecha_de_apertura_efectiva",
            "proveedores_invitados", "proveedores_con_invitacion", "visualizaciones_del",
            "proveedores_que_manifestaron", "respuestas_al_procedimiento", "respuestas_externas",
            "conteo_de_respuestas_a_ofertas", "proveedores_unicos_con", "numero_de_lotes",
            "estado_del_procedimiento", "adjudicado", "fecha_adjudicacion", "valor_total_adjudicacion",
            "subtipo_de_contrato", "estado_resumen", "fase", "nombre_del_procedimiento"]

session = requests.Session()


class Throttled(RuntimeError):
    pass


def soql(did, params, fmt="json"):
    """Consulta SoQL con reintentos. Se detiene si el portal limita el uso (403/429)."""
    url = f"{BASE}/resource/{did}.{fmt}"
    for intento in range(4):
        try:
            r = session.get(url, params=params, timeout=300)
        except requests.RequestException:
            time.sleep(5 * (intento + 1))
            continue
        if r.status_code in (403, 429):
            raise Throttled(f"{r.status_code} en {did}: {r.text[:200]}")
        if r.status_code == 200:
            if fmt == "json":
                return r.json()
            texto = r.content.decode("utf-8")
            if not texto.strip():
                return pd.DataFrame()
            return pd.read_csv(io.StringIO(texto), dtype=str, low_memory=False)
        time.sleep(5 * (intento + 1))
    raise RuntimeError(f"Fallo {did} (ultimo status {r.status_code})")


def in_list(valores):
    return ",".join("'" + str(v).replace("'", "''") + "'" for v in valores)


# ---------- 1. muestra de Contratos estratificada por mes ----------
def conteo_diario_poblacion():
    """Una sola consulta: contratos por dia de firma desde 2018 (se uso para repartir las cuotas)."""
    diario = pd.DataFrame(soql(DS["contratos"], {
        "$select": "date_trunc_ymd(fecha_de_firma) as d, count(*) as n",
        "$where": "fecha_de_firma >= '2018-01-01T00:00:00'",
        "$group": "d", "$order": "d", "$limit": 50000}))
    diario["d"] = pd.to_datetime(diario["d"])
    diario["n"] = diario["n"].astype(int)
    diario["mes"] = diario.d.dt.to_period("M")
    diario["ventana"] = pd.cut(diario.d.dt.day, [0, 7, 14, 21, 31], labels=False)
    return diario.groupby(["mes", "ventana"]).agg(n=("n", "sum"), d0=("d", "min"), d1=("d", "max")).reset_index()


def asignar_cuotas(ventanas, cuota=CUOTA_MES):
    """Reparte la cuota del mes entre las 4 ventanas, proporcional al volumen (metodo del mayor resto)."""
    ventanas = ventanas.copy()
    ventanas["cuota"] = 0
    for mes, g in ventanas.groupby("mes"):
        if g.n.sum() <= cuota:
            ventanas.loc[g.index, "cuota"] = g.n
            continue
        exacta = g.n / g.n.sum() * cuota
        base = np.floor(exacta).astype(int)
        resto = cuota - base.sum()
        extra = (exacta - base).sort_values(ascending=False).index[:resto]
        base.loc[extra] += 1
        ventanas.loc[g.index, "cuota"] = base
    return ventanas


def descargar_mes(mes, ventanas, sel_contratos, destino):
    """Descarga y submuestrea un mes. En cada ventana se filtra por el ultimo digito de id_contrato
    (en vez de $order + $limit, que sesga hacia el inicio) y se submuestrea localmente."""
    f = destino / f"{mes}.parquet"
    if f.exists():
        return f
    partes = []
    r_local = np.random.default_rng([SEED, int(mes.year * 100 + mes.month)])
    for _, v in ventanas[ventanas.mes == mes].iterrows():
        if v.cuota == 0:
            continue
        g = int(np.clip(math.ceil(10 * v.cuota * 1.3 / v.n), 1, 10))
        where = f"fecha_de_firma between '{v.d0:%Y-%m-%d}T00:00:00' and '{v.d1:%Y-%m-%d}T23:59:59'"
        if g < 10:
            digitos = r_local.choice(10, size=g, replace=False)
            where += " AND (" + " OR ".join(f"id_contrato like '%{d}'" for d in digitos) + ")"
        df = soql(DS["contratos"], {"$select": ",".join(sel_contratos), "$where": where, "$limit": int(v.n) + 100}, fmt="csv")
        if len(df) > v.cuota:
            df = df.iloc[np.sort(r_local.choice(len(df), size=int(v.cuota), replace=False))]
        partes.append(df)
    out = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=sel_contratos)
    out.to_parquet(f)
    return f


# ---------- 2. datasets unidos, solo para los contratos de la muestra ----------
def por_lotes(valores, fn, workers=3):
    lotes = [valores[i:i + LOTE] for i in range(0, len(valores), LOTE)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(fn, lotes))
    res = [r for r in res if len(r)]
    return pd.concat(res, ignore_index=True) if res else pd.DataFrame()


def bajar_procesos(lote):
    return soql(DS["procesos"], {"$select": ",".join(SEL_PROC), "$where": f"id_del_portafolio in ({in_list(lote)})", "$limit": 5000}, fmt="csv")


def bajar_adiciones(lote):
    """Agregado en el servidor: una fila por contrato."""
    sel = ("id_contrato, count(*) as n_eventos, "
           "sum(case(tipo='Adicion en el valor',1,true,0)) as n_adic_valor, "
           "sum(case(tipo='Extension',1,true,0)) as n_extension, "
           "sum(case(tipo='Modificacion General',1,true,0)) as n_mod_general, "
           "min(fecharegistro) as primer_evento, max(fecharegistro) as ultimo_evento")
    r = soql(DS["adiciones"], {"$select": sel, "$where": f"id_contrato in ({in_list(lote)})", "$group": "id_contrato", "$limit": 5000})
    return pd.DataFrame(r)


def bajar_ejecucion(lote):
    sel = "identificadorcontrato as id_contrato, count(*) as n_filas_ejec, max(porcentaje_de_avance_real) as avance_real_max"
    r = soql(DS["ejecucion"], {"$select": sel, "$where": f"identificadorcontrato in ({in_list(lote)})", "$group": "identificadorcontrato", "$limit": 5000})
    return pd.DataFrame(r)


def bajar_modificaciones(lote):
    """Solo para la submuestra exploratoria de 3.000 contratos (el dataset completo tiene ~30 M de filas)."""
    sel = "id_contrato, valor_modificacion, dias_extendidos, estado_modificacion, numero_version, fecha_creacion"
    return soql(DS["modificaciones"], {"$select": sel, "$where": f"id_contrato in ({in_list(lote)})", "$limit": 20000}, fmt="csv")
