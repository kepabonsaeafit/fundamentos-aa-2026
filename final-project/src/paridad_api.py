"""Prueba de paridad API frente a base (G1): las features que arma la app desde la API son las del entrenamiento.

Toma contratos de la base local (3 por anio de firma, 2019-2025), los pide a datos.gov.co y compara, feature por
feature, construir_features(API) contra construir_features(base). Es un script de verificacion, no un test automatico:
usa la red y la base local. Cuesta 2 peticiones (una a Contratos y otra a Procesos, por lotes).

Uso (desde final-project/):  python -m src.paridad_api [--por-anio 3]
Escribe resultados/paridad_api_base.md con el resumen (y la tabla de diferencias, si las hay).
"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from . import api_secop as A
from . import config
from . import features as F
from . import modelo_io, uniones


def _comillas(ids):
    return ", ".join(f"'{i}'" for i in ids)


def pedir_lote(cliente, ids):
    """Contratos y procesos de varios contratos en 2 peticiones (solo ids con formato seguro)."""
    for i in ids:
        A._donde_igual("id_contrato", i)
    contratos = pd.DataFrame(cliente.consultar(config.DATASETS["contratos"], {
        "$select": ",".join(A.COLS_CONTRATO_API), "$where": f"id_contrato in ({_comillas(ids)})", "$limit": len(ids) * 2}))
    portafolios = sorted(contratos["proceso_de_compra"].dropna().unique())
    for p in portafolios:
        A._donde_igual("id_del_portafolio", p)
    procesos = pd.DataFrame(cliente.consultar(config.DATASETS["procesos"], {
        "$select": ",".join(A.COLS_PROCESO_API), "$where": f"id_del_portafolio in ({_comillas(portafolios)})",
        "$limit": 5000}))
    return contratos, procesos


def features_api(contratos, procesos):
    unido = uniones.unir_contrato_proceso(contratos.drop_duplicates("id_contrato").reset_index(drop=True),
                                          uniones.elegir_proceso(procesos))
    return F.construir_features(unido).set_index(unido["id_contrato"].to_numpy())


def comparar(x_api, x_base):
    """Una fila por (contrato, feature) que difiere. Dos NaN cuentan como iguales; los numeros con isclose."""
    filas = []
    for idc in x_base.index:
        if idc not in x_api.index:
            filas.append({"id_contrato": idc, "feature": "(contrato)", "api": "no devuelto por la API", "base": ""})
            continue
        for col in F.FEATURES:
            a, b = x_api.at[idc, col], x_base.at[idc, col]
            if pd.isna(a) and pd.isna(b):
                continue
            if not pd.isna(a) and not pd.isna(b) and (a == b or (isinstance(a, float) and np.isclose(a, b))):
                continue
            filas.append({"id_contrato": idc, "feature": col, "api": a, "base": b})
    return pd.DataFrame(filas, columns=["id_contrato", "feature", "api", "base"])


def main(por_anio=3, semilla=config.SEED):
    base = pd.read_parquet(config.DATA_PROCESSED / "base_contratos.parquet")
    elegibles = base[base["eval_en_procesos"]]
    muestra = elegibles.groupby("anio_firma", group_keys=False).sample(n=por_anio, random_state=semilla)
    x_base = F.construir_features(muestra).set_index(muestra["id_contrato"].to_numpy())

    cliente = A.crear_cliente()
    contratos, procesos = pedir_lote(cliente, muestra["id_contrato"].tolist())
    x_api = features_api(contratos, procesos)
    difs = comparar(x_api, x_base)

    n, n_celdas = len(x_base), len(x_base) * len(F.FEATURES)
    print(f"Peticiones: {cliente.peticiones} | contratos pedidos: {n} | devueltos por la API: {x_api.index.isin(x_base.index).sum()}")
    print(f"Celdas comparadas: {n_celdas} | distintas: {len(difs)} ({len(difs) / n_celdas:.2%})")
    if len(difs):
        print("\nDiferencias por feature:")
        print(difs.groupby("feature").size().sort_values(ascending=False).to_string())
        print("\nPrimeras diferencias:")
        print(difs.head(15).to_string(index=False))
    try:   # el puntaje del dummy con las dos fuentes: lo que importa para el usuario
        pipe, _ = modelo_io.cargar_modelo(config.MODELS / "modelo_prorroga_dummy.joblib")
        comunes = x_base.index.intersection(x_api.index)
        d = np.abs(pipe.predict_proba(x_api.loc[comunes])[:, 1] - pipe.predict_proba(x_base.loc[comunes])[:, 1])
        print(f"\nPuntaje del modelo dummy, API frente a base: diferencia maxima {d.max():.2e}, media {d.mean():.2e}")
    except Exception as e:      # el modelo es opcional para esta verificacion
        print("\n(no se pudo comparar el puntaje del modelo:", type(e).__name__, ")")
    destino = config.RAIZ / "resultados" / "paridad_api_base.md"
    destino.parent.mkdir(exist_ok=True)
    lineas = [
        "# Paridad API frente a base",
        "",
        f"- Fecha de la prueba: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Contratos comparados: {n} ({por_anio} por anio de firma, 2019-2025, semilla {semilla})",
        f"- Celdas comparadas (contratos x {len(F.FEATURES)} features): {n_celdas}",
        f"- Celdas distintas: {len(difs)}",
        f"- Peticiones a datos.gov.co: {cliente.peticiones}",
        "",
        "Compara `construir_features` aplicada a lo que devuelve la API contra la aplicada a las filas de la base v1.",
    ]
    if len(difs):
        lineas += ["", "## Diferencias", "", "| id_contrato | feature | api | base |", "|---|---|---|---|"]
        lineas += [f"| {r.id_contrato} | {r.feature} | {r.api} | {r.base} |" for r in difs.itertuples()]
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("escrito", destino.relative_to(config.RAIZ))
    return difs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-anio", type=int, default=3)
    main(ap.parse_args().por_anio)
