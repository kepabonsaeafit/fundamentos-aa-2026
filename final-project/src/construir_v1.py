"""Construye la base v1 (PLAN.md F3) en dos modos con el mismo codigo, sin red.

  Modo A: 10 % por digito 4 (F2 normal, autoponderada). Entrada data/raw/f2/, salida data/processed/v1_A/. peso_mes = 1,0.
  Modo B: muestra de la exploracion (2.000 contratos por mes, no autoponderada). Entrada data/raw/f2_b/, salida
          data/processed/v1_B/. peso_mes = N_mes / n_mes.

Cada entrada tiene el formato de src/descarga.py: contratos/anio=AAAA/mes=MM_*.parquet, procesos/lote_*.parquet,
modificaciones/lote_*.parquet, _manifiesto.csv y completitud.json. Sin red: solo lee archivos locales.

Etapas: carga de unidades -> join con Procesos y tipado (src/base.py, src/uniones.py) -> etiqueta (src/etiqueta.py) -> pesos
(src/pesos.py) -> features (src/features.py) -> escritura + sha256 + relectura -> reporte de calidad (src/calidad.py) -> meta.
Un latido en el log cada 30 s y el tiempo y la RAM pico de cada etapa en el meta.

Uso (desde final-project/, con .venv/Scripts/python.exe):
  python -m src.construir_v1 --modo A      # lee data/raw/f2/   -> data/processed/v1_A/
  python -m src.construir_v1 --modo B      # lee data/raw/f2_b/ -> data/processed/v1_B/
Codigos de salida: 0 todo bien; 1 la base se escribio pero algun chequeo de calidad bloqueante fallo (ver reporte_calidad.md);
2 entrada incompleta o ilegible (no se escribe nada).
Para publicar la base elegida, copiar de v1_<modo>/ a data/processed/: base_contratos.parquet, base_contratos.meta.json y
reporte_calidad.md (el sha256 no cambia con la copia).
"""
import argparse
import glob
import hashlib
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from . import base as B
from . import config
from . import etiqueta as E
from . import features as F
from . import pesos as P

VERSION = "v1"
COLUMNAS_PESOS = B.COLUMNAS_PESOS
COLUMNAS_BASE_V1 = B.COLUMNAS_BASE + COLUMNAS_PESOS
RUTAS = {"A": (config.DATA_RAW / "f2", config.DATA_PROCESSED / "v1_A"),
         "B": (config.DATA_RAW / "f2_b", config.DATA_PROCESSED / "v1_B")}
ARCHIVOS_CODIGO = ["construir_v1.py", "etiqueta.py", "pesos.py", "calidad.py", "base.py", "features.py", "uniones.py", "config.py"]
LATIDO_S = 30
TEXTO = pd.StringDtype(na_value=np.nan)

COLS_CONTRATOS = list(dict.fromkeys(["id_contrato", "proceso_de_compra", "nombre_entidad", "dias_adicionados"] + F.COLS_CONTRATO))
COLS_PROCESOS = list(dict.fromkeys(["id_del_portafolio", "id_del_proceso", "adjudicado", "fecha_adjudicacion",
                                    "valor_total_adjudicacion"] + [c[2:] for c in F.COLS_PROCESO]))


class Registro:
    """Log a archivo y pantalla, con un latido cada LATIDO_S segundos y el tiempo y la RAM pico de cada etapa."""

    def __init__(self, ruta_log):
        ruta_log.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(ruta_log, "a", encoding="utf-8", buffering=1)
        self.t0 = time.perf_counter()
        self.etapa_actual = "inicio"
        self.etapas = {}
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._latir, daemon=True)
        self._hilo.start()

    @staticmethod
    def ram_mb():
        mi = psutil.Process().memory_info()
        return round(mi.rss / 2 ** 20), round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20)

    def log(self, msg):
        linea = f"{datetime.now():%H:%M:%S} {msg}"
        print(linea, flush=True)
        self.f.write(linea + "\n")

    def _latir(self):
        while not self._parar.wait(LATIDO_S):
            actual, pico = self.ram_mb()
            self.log(f"LATIDO etapa={self.etapa_actual} t={time.perf_counter() - self.t0:.0f}s RAM={actual}MB pico={pico}MB")

    def etapa(self, nombre):
        return _Etapa(self, nombre)

    def cerrar(self):
        self._parar.set()
        self.log("FIN")
        self.f.close()


class _Etapa:
    def __init__(self, reg, nombre):
        self.reg, self.nombre = reg, nombre

    def __enter__(self):
        self.reg.etapa_actual = self.nombre
        self.t = time.perf_counter()
        self.reg.log(f"[{self.nombre}] inicio")
        return self

    def __exit__(self, tipo, valor, tb):
        seg = time.perf_counter() - self.t
        actual, pico = self.reg.ram_mb()
        self.reg.etapas[self.nombre] = {"segundos": round(seg, 2), "ram_mb": actual, "ram_pico_mb": pico}
        self.reg.log(f"[{self.nombre}] {'fin' if tipo is None else 'ERROR ' + tipo.__name__} {seg:.1f} s RAM {actual} MB (pico {pico} MB)")


# ---------------------------------------------------------------- lectura de las unidades de f2
def leer_leyendo(reg, archivos, columnas, nombre):
    """Concatena los parquet de una carpeta de f2 (todo texto) con las columnas pedidas; las que falten quedan NaN."""
    if not archivos:
        raise FileNotFoundError(f"no hay archivos de {nombre}")
    presentes = [c for c in columnas if c in pq.ParquetFile(archivos[0]).schema_arrow.names]
    partes = []
    for i, f in enumerate(archivos, 1):
        d = pd.read_parquet(f, columns=presentes)
        partes.append(d.astype({c: TEXTO for c in d.columns if d[c].dtype == object or str(d[c].dtype) == "string"}))
        if i % 200 == 0:
            reg.log(f"  {nombre}: {i}/{len(archivos)} archivos")
    df = pd.concat(partes, ignore_index=True).reindex(columns=columnas)
    return df.astype({c: TEXTO for c in df.columns if df[c].dtype == object})


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8")) if Path(ruta).exists() else {}


def corte_de(origen, completitud):
    """Corte de la descarga (hora local sin zona). Si viene con zona, se pasa a hora de Colombia."""
    crudo = completitud.get("corte") or leer_json(origen / "_estado.json").get("corte")
    if not crudo:
        raise ValueError(f"{origen}: no encuentro el corte en completitud.json ni en _estado.json")
    t = pd.Timestamp(crudo)
    if t.tzinfo is not None:
        t = t.tz_convert("America/Bogota").tz_localize(None)
    return t


def hash_codigo():
    raiz = config.RAIZ
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=raiz, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    fuentes = {n: hashlib.sha256((raiz / "src" / n).read_bytes()).hexdigest() for n in ARCHIVOS_CODIGO if (raiz / "src" / n).exists()}
    return {"commit": git("rev-parse", "HEAD") or None, "rama": git("rev-parse", "--abbrev-ref", "HEAD") or None,
            "arbol_con_cambios_en_src": bool(git("status", "--porcelain", "--", "src")),
            "sha256_fuentes": fuentes,
            "sha256_codigo": hashlib.sha256("".join(f"{k}:{v}" for k, v in sorted(fuentes.items())).encode()).hexdigest()}


def sha256_archivo(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 22), b""):
            h.update(bloque)
    return h.hexdigest()


# ---------------------------------------------------------------- construccion
def construir(modo, origen, destino, conteo_exacto=None, permitir_incompleto=False, refiltrar_corte=False,
              calidad=True, reg=None, dirs=None):
    origen, destino = Path(origen), Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    propio = reg is None
    reg = reg or Registro(destino / "_construir_v1.log")
    try:
        return _construir(modo, origen, destino, conteo_exacto, permitir_incompleto, refiltrar_corte, calidad, reg, dirs or {})
    finally:
        if propio:
            reg.cerrar()


def _construir(modo, origen, destino, conteo_exacto, permitir_incompleto, refiltrar_corte, calidad, reg, dirs):
    # dirs permite leer una de las tres carpetas de otro lugar (solo lectura), p. ej. los contratos reales de f2 en una prueba de tiempos
    carpeta = {k: Path(dirs[k]) if dirs.get(k) else origen / k for k in ("contratos", "procesos", "modificaciones")}
    reg.log(f"construir_v1 modo {modo}: origen {origen} -> destino {destino}")
    completitud = leer_json(origen / "completitud.json")
    completitud_ok = bool(completitud.get("ok"))
    if not completitud_ok and not permitir_incompleto:
        reg.log(f"ERROR: {origen / 'completitud.json'} no existe o ok != true; use --permitir-incompleto solo para pruebas")
        return None, 2
    corte = corte_de(origen, completitud)
    fecha_corte = corte.normalize()
    ctx = {"modo": modo, "origen": str(origen), "destino": str(destino), "corte": str(corte),
           "fecha_corte_censura": str(fecha_corte.date()), "completitud_ok": completitud_ok,
           "completitud_motivo": completitud.get("motivo"), "digito": completitud.get("digito"),
           "columnas_leidas": {"contratos": COLS_CONTRATOS, "procesos": COLS_PROCESOS, "modificaciones": E.COLS_MODIF}}

    with reg.etapa("carga_contratos"):
        arch = sorted(glob.glob(str(carpeta["contratos"] / "anio=*" / "mes=*.parquet")))
        contratos = leer_leyendo(reg, arch, COLS_CONTRATOS, "contratos")
        n0 = len(contratos)
        contratos = contratos.drop_duplicates()
        ctx["contratos"] = {"archivos": len(arch), "filas_leidas": n0, "duplicados_exactos": n0 - len(contratos)}
        if not contratos["id_contrato"].is_unique:
            raise ValueError("Contratos tiene id_contrato repetidos con filas distintas: revisar la descarga")
    with reg.etapa("carga_procesos"):
        arch = sorted(glob.glob(str(carpeta["procesos"] / "lote_*.parquet")))
        procesos = leer_leyendo(reg, arch, COLS_PROCESOS, "procesos")
        por_portafolio = procesos["id_del_portafolio"].value_counts()
        ctx["procesos"] = {"archivos": len(arch), "filas": int(len(procesos)), "portafolios": int(len(por_portafolio)),
                           "portafolios_con_varias_filas": int((por_portafolio > 1).sum())}
    with reg.etapa("carga_modificaciones"):
        arch = sorted(glob.glob(str(carpeta["modificaciones"] / "lote_*.parquet")))
        modif = leer_leyendo(reg, arch, E.COLS_MODIF, "modificaciones")
        ctx["modificaciones_archivos"] = len(arch)

    with reg.etapa("join_y_tipado"):
        n_entrada = len(contratos)
        base = B.construir_base(contratos, procesos, fecha_corte)      # join sin duplicar (uniones.py), tipado, censura 4.5
        ctx["join"] = {"contratos_entrada": int(n_entrada), "filas_base": int(len(base)),
                       "fuera_de_2019_2025_o_sin_fecha": int(n_entrada - len(base))}
        ctx["proceso_de_compra_nulos"] = int(contratos["proceso_de_compra"].isna().sum())
        del contratos, procesos
    with reg.etapa("etiqueta"):
        prep, info_m = E.preparar_modificaciones(modif, corte=corte, aplicar_corte=refiltrar_corte)
        base, aportes = E.etiquetar(base, prep)
        ctx["modificaciones"] = info_m
        ctx["aportes_etiqueta"] = aportes
        del modif, prep
    with reg.etapa("pesos"):
        base, tabla_pesos, meta_pesos = P.asignar_pesos(base, modo, conteo_exacto)
        base = base.sort_values(["fecha_de_firma", "id_contrato"], kind="mergesort").reset_index(drop=True)
        base = base[COLUMNAS_BASE_V1]
        tabla_pesos.to_csv(destino / "peso_mes_tabla.csv", index_label="mes")
        ctx["pesos"] = meta_pesos
    with reg.etapa("validacion_esquema"):
        B.validar_esquema(base)          # incluye las tres columnas de pesos al final
    with reg.etapa("features"):
        X = F.construir_features(base)
    with reg.etapa("escritura_parquet"):
        ruta = destino / "base_contratos.parquet"
        base.to_parquet(ruta, index=False)
    with reg.etapa("sha256_y_relectura"):
        sha = sha256_archivo(ruta)
        relectura = pd.read_parquet(ruta)
        pd.testing.assert_frame_equal(base, relectura, check_dtype=True)
        sha_dos = sha256_archivo(ruta)
        if sha != sha_dos:
            raise RuntimeError("el sha256 cambio entre dos lecturas del mismo archivo")
        ctx["archivo"] = {"nombre": ruta.name, "bytes": ruta.stat().st_size, "sha256": sha,
                          "relectura_igual_a_memoria": True, "sha256_verificado_dos_veces": True}
        del relectura

    resultado = None
    if calidad:
        from . import calidad as Q
        with reg.etapa("calidad"):
            ctx["etapas_previas"] = dict(reg.etapas)
            resultado = Q.evaluar(base, X, ctx)
    with reg.etapa("meta_y_reportes"):
        meta = armar_meta(modo, base, ctx, tabla_pesos, resultado, reg)
        (destino / "base_contratos.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        if resultado is not None:
            Q.escribir(resultado, destino, reg.etapas)
    reg.etapas["total"] = {"segundos": round(time.perf_counter() - reg.t0, 1), "ram_pico_mb": reg.ram_mb()[1]}
    meta["rendimiento"] = reg.etapas
    (destino / "base_contratos.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if resultado is not None:
        Q.escribir(resultado, destino, reg.etapas)
    reg.log(f"base {modo}: {len(base)} filas, sha256 {sha}, calidad: {resultado['resumen'] if resultado else 'omitida'}")
    bloqueantes = resultado["resumen"]["fallas_bloqueantes"] if resultado else 0
    return base, (1 if bloqueantes else 0)


def armar_meta(modo, base, ctx, tabla_pesos, resultado, reg):
    por_anio = base.groupby("anio_firma").agg(filas=("y", "size"), tasa_y=("y", "mean"), tasa_y_dias=("y_dias", "mean"),
                                              madura=("eval_madura", "mean"))
    w = base["peso_mes"]
    p = ctx["pesos"]
    autoponderada = modo == "A"
    return {
        "version": VERSION,
        "modo": modo,
        "base": f"v1_{modo}",
        "features_version": F.FEATURES_VERSION,
        "muestra": ("10 % por ultimo digito de id_contrato (digito " + str(ctx.get("digito") or "4") + "), 2019-2025; autoponderada"
                    if autoponderada else
                    "muestreo estratificado por mes con asignacion igual (2.000 contratos por mes), foto del 14-oct-2026, 2019-2025; "
                    "NO autoponderada: usar peso_mes en las metricas de evaluacion"),
        "autoponderada": autoponderada,
        "fechas_de_corte": {
            "corte_de_la_descarga": ctx["corte"],
            "nota_zona": "hora local de Colombia, sin zona, tal como la guarda descarga.py. El corte de Modificaciones (fecha_version <= corte) "
                         "se aplico en el servidor. En B1 se vio que fecha_version es calendar_date (flotante) y viene a medianoche, asi que el corte "
                         "solo cuenta como fecha y no se convierte a UTC; en A, si el campo estuviera en UTC, el corte efectivo queda hasta 5 h antes "
                         "del arranque (limitacion aceptada). Ver docs/BASE_V1.md",
            "fecha_de_corte_de_la_censura": ctx["fecha_corte_censura"],
        },
        "hash_codigo": hash_codigo(),
        "origen": {"carpeta": ctx["origen"], "completitud_ok": ctx["completitud_ok"], "motivo": ctx["completitud_motivo"]},
        "etiqueta": "y = y_dias | y_modif. y_dias: dias_adicionados > 0. y_modif: existe una modificacion con estado 'Publicado', "
                    "dias_extendidos > 0 y fecha_de_aprobacion > fecha_de_firma + 1 dia. Objetivo: ampliacion del plazo "
                    "(suspensiones y ajustes de fecha cuentan como positivos, ~2,4 % de los positivos). Filas exactas de "
                    "Modificaciones deduplicadas.",
        "madurez": f"eval_madura: fin planeado (inicio o firma + duracion del proceso, sin duraci_n_del_contrato) + {B.DIAS_MADUREZ} dias "
                   "<= fecha de corte de la censura; si la duracion planeada es 0 o falta, madura si fecha_de_firma + 365 dias <= fecha de corte",
        "conteos": {
            "filas": int(len(base)),
            "contratos_unicos": int(base["id_contrato"].nunique()),
            "positivos_y": int(base["y"].sum()), "positivos_y_dias": int(base["y_dias"].sum()),
            "positivos_y_modif": int(base["y_modif"].sum()),
            "tasa_y": round(float(base["y"].mean()), 4),
            "tasa_y_ponderada": round(float(np.average(base["y"], weights=w)), 4),
            "contratos": ctx["contratos"], "procesos": ctx["procesos"], "join": ctx["join"],
            "modificaciones": ctx["modificaciones"], "aportes_etiqueta": ctx["aportes_etiqueta"],
            "por_anio": {int(a): {"filas": int(r.filas), "tasa_y": round(float(r.tasa_y), 4),
                                  "tasa_y_dias": round(float(r.tasa_y_dias), 4), "madura": round(float(r.madura), 4)}
                         for a, r in por_anio.iterrows()},
        },
        "pesos": {
            "peso_mes": p["formula"],
            "fuente_N_mes": p["fuente_N_mes"],
            "como_se_calculo": ("peso_mes = N_mes / n_mes. N_mes: contratos de la poblacion firmados en el mes (ver fuente_N_mes). "
                                "n_mes: filas de la base en ese mes tras deduplicar." if modo == "B" else
                                "peso_mes = 1,0 para todas las filas; N_mes solo se llena si hay conteo mensual exacto; n_mes = filas de la base en el mes."),
            "error_esperado": p.get("error_esperado", "no aplica (peso constante)"),
            "resumen": {"min": float(w.min()), "mediana": float(w.median()), "max": float(w.max()),
                        "suma": float(w.sum()), "tamano_efectivo_kish": round(P.kish_efectivo(w), 1)},
            "nota": ("En B la base NO es autoponderada. El entrenamiento es sin pesos (decision del usuario); las metricas de evaluacion "
                     "se reportan ponderadas con peso_mes y tambien sin pesos; la ablacion de J3 con pesos recortados es obligatoria."
                     if modo == "B" else
                     "Base autoponderada: peso_mes es constante, asi que la evaluacion ponderada da lo mismo que la cruda; la columna "
                     "existe para que el codigo de J1 funcione igual con A y con B."),
        },
        "columnas": COLUMNAS_BASE_V1,
        "dtypes": {c: str(t) for c, t in base.dtypes.items()},
        "archivo": ctx["archivo"],
        "sha256": ctx["archivo"]["sha256"],
        "calidad": resultado["resumen"] if resultado else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modo", required=True, choices=["A", "B"])
    ap.add_argument("--origen", help="carpeta de entrada (por defecto data/raw/f2 en A, data/raw/f2_b en B)")
    ap.add_argument("--destino", help="carpeta de salida (por defecto data/processed/v1_A o v1_B)")
    ap.add_argument("--conteo-mensual", help="CSV de conteo mensual exacto (por defecto data/raw/f2/conteo_mensual_poblacion.csv)")
    ap.add_argument("--permitir-incompleto", action="store_true", help="solo pruebas: sigue aunque completitud.json no diga ok")
    ap.add_argument("--refiltrar-corte", action="store_true", help="descarta Modificaciones con fecha_version > corte (por defecto solo las cuenta)")
    ap.add_argument("--sin-calidad", action="store_true", help="solo para medir tiempos")
    for k in ("contratos", "procesos", "modificaciones"):
        ap.add_argument(f"--dir-{k}", help=f"solo pruebas: lee {k} de esta carpeta (solo lectura) en vez de <origen>/{k}")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    origen, destino = RUTAS[a.modo]
    origen, destino = Path(a.origen) if a.origen else origen, Path(a.destino) if a.destino else destino
    try:
        dirs = {k: getattr(a, f"dir_{k}") for k in ("contratos", "procesos", "modificaciones")}
        _, codigo = construir(a.modo, origen, destino, a.conteo_mensual, a.permitir_incompleto, a.refiltrar_corte, not a.sin_calidad, dirs=dirs)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR de entrada: {e}", file=sys.stderr)
        return 2
    return codigo


if __name__ == "__main__":
    sys.exit(main())
