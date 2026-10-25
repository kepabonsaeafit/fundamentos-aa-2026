"""Funciones reutilizables del EDA (F4).

El notebook `notebooks/03_eda.ipynb` llama estas funciones. Todo lo que depende de la etiqueta
recibe `col_y` como parametro para que el mismo codigo corra con la base v0 (y = y_dias) y con la v1
(etiqueta definitiva). Las funciones de calidad, distribucion y deriva no leen la etiqueta.

Convenciones:
- Las tablas se devuelven como DataFrame; las graficas se dibujan sobre un `ax` o crean su figura.
- Los intervalos de proporcion son de Wilson (95 %). Son descriptivos: no corrigen la correlacion
  dentro de una entidad; para H6 se reporta ademas el bootstrap por entidad en el notebook.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


# ------------------------------------------------------------------ carga y utilidades
def cargar_base(ruta=None):
    """Carga la base procesada y su metadata. Devuelve (base, meta)."""
    ruta = Path(ruta) if ruta is not None else config.DATA_PROCESSED / "base_contratos.parquet"
    meta_ruta = ruta.with_suffix(".meta.json")
    base = pd.read_parquet(ruta)
    meta = json.loads(meta_ruta.read_text(encoding="utf-8")) if meta_ruta.exists() else {}
    return base, meta


def prefijo_provisional(meta):
    """Marca para los resultados que dependen de la etiqueta mientras se trabaja con la v0."""
    return "[PROVISIONAL v0] " if str(meta.get("version", "")).lower() == "v0" else ""


def config_graficas():
    """Estilo sobrio y consistente para el notebook."""
    plt.rcParams.update({
        "figure.figsize": (10, 4),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
    })


# ------------------------------------------------------------------ calidad de datos
def resumen_faltantes(df, columnas=None):
    """Una fila por columna: n, faltantes, porcentaje y numero de valores unicos."""
    columnas = list(columnas) if columnas is not None else list(df.columns)
    filas = []
    for c in columnas:
        s = df[c]
        filas.append({
            "columna": c,
            "tipo": str(s.dtype),
            "n": len(s),
            "faltantes": int(s.isna().sum()),
            "pct_faltantes": 100 * float(s.isna().mean()),
            "n_unicos": int(s.nunique(dropna=True)),
        })
    return (pd.DataFrame(filas)
            .sort_values(["pct_faltantes", "columna"], ascending=[False, True])
            .reset_index(drop=True))


def graficar_faltantes(tabla, top=20, titulo="Faltantes por columna"):
    """Barras horizontales de porcentaje de faltantes (solo columnas con faltantes > 0)."""
    t = tabla[tabla["faltantes"] > 0].head(top).sort_values("pct_faltantes")
    fig, ax = plt.subplots(figsize=(8, max(2.2, 0.28 * len(t) + 1)))
    if t.empty:
        ax.text(0.5, 0.5, "Sin faltantes", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.barh(t["columna"], t["pct_faltantes"], color="#4c72b0")
        for y, (_, r) in enumerate(t.iterrows()):
            ax.text(r["pct_faltantes"], y, f" {r['pct_faltantes']:.2f}% ({r['faltantes']:,})",
                    va="center", fontsize=7)
        ax.set_xlabel("% de filas con faltante")
        ax.set_xlim(0, max(t["pct_faltantes"].max() * 1.35, 0.5))
    ax.set_title(titulo)
    plt.tight_layout()
    return fig, ax


def _fila_chequeo(chequeo, mascara, detalle=""):
    n = int(np.asarray(mascara).sum())
    return {"chequeo": chequeo, "n": n, "pct": 100 * n / len(mascara), "detalle": detalle}


def revisar_valores_imposibles(base):
    """Lista de chequeos binarios de calidad. No usa la etiqueta salvo para validar sus valores."""
    filas = []
    filas.append(_fila_chequeo("id_contrato duplicado", base["id_contrato"].duplicated(keep=False)))
    filas.append(_fila_chequeo("fecha_de_firma nula", base["fecha_de_firma"].isna()))
    filas.append(_fila_chequeo("anio_firma fuera de 2019-2025", ~base["anio_firma"].between(2019, 2025)))
    filas.append(_fila_chequeo("fecha_de_inicio nula", base["fecha_de_inicio_del_contrato"].isna()))
    dias_inicio = (base["fecha_de_inicio_del_contrato"] - base["fecha_de_firma"]).dt.days
    filas.append(_fila_chequeo("inicio mas de 365 dias antes de la firma", dias_inicio < -365))
    filas.append(_fila_chequeo("inicio mas de 365 dias despues de la firma", dias_inicio > 365))
    filas.append(_fila_chequeo("p_fecha_de_publicacion nula", base["p_fecha_de_publicacion_del"].isna()))
    dias_pub = (base["fecha_de_firma"] - base["p_fecha_de_publicacion_del"]).dt.days
    filas.append(_fila_chequeo("publicacion posterior a la firma", dias_pub < 0))
    filas.append(_fila_chequeo("p_duracion nula", base["p_duracion"].isna()))
    filas.append(_fila_chequeo("p_duracion <= 0", base["p_duracion"] <= 0))
    dur = base["eval_dur_planeada_dias"]
    filas.append(_fila_chequeo("duracion planeada nula", dur.isna()))
    filas.append(_fila_chequeo("duracion planeada <= 0", dur <= 0))
    filas.append(_fila_chequeo("duracion planeada > 3650 dias", dur > 3650))
    filas.append(_fila_chequeo("p_precio_base nulo", base["p_precio_base"].isna()))
    filas.append(_fila_chequeo("p_precio_base negativo", base["p_precio_base"] < 0))
    filas.append(_fila_chequeo("p_precio_base igual a cero", base["p_precio_base"] == 0))
    filas.append(_fila_chequeo("valor_de_pago_adelantado negativo", base["valor_de_pago_adelantado"] < 0))
    filas.append(_fila_chequeo("p_numero_de_lotes negativo", base["p_numero_de_lotes"] < 0))
    filas.append(_fila_chequeo("p_numero_de_lotes igual a cero", base["p_numero_de_lotes"] == 0))
    for c in ["p_proveedores_invitados", "p_proveedores_con_invitacion", "p_proveedores_que_manifestaron",
              "p_respuestas_al_procedimiento", "p_respuestas_externas", "p_conteo_de_respuestas_a_ofertas",
              "p_proveedores_unicos_con"]:
        filas.append(_fila_chequeo(f"{c} negativo", base[c] < 0))
    filas.append(_fila_chequeo("y fuera de {0,1}", ~base["y"].isin([0, 1])))
    filas.append(_fila_chequeo("y_dias fuera de {0,1}", ~base["y_dias"].isin([0, 1])))
    filas.append(_fila_chequeo("fin planeado anterior a la firma", base["eval_fin_planeado"] < base["fecha_de_firma"]))
    pii = sorted(set(base.columns) & config.PII)
    filas.append({"chequeo": "PII en la base", "n": len(pii), "pct": np.nan,
                  "detalle": ", ".join(pii) if pii else "ninguna"})
    from . import features as F
    fuga = (set(base.columns) & F.COLUMNAS_FUGA) - set(F._ETIQUETAS)
    filas.append({"chequeo": "columnas de fuga en la base", "n": len(fuga), "pct": np.nan,
                  "detalle": ", ".join(sorted(fuga)) if fuga else "ninguna"})
    manifestaron = base["p_proveedores_que_manifestaron"].dropna()
    constante = len(manifestaron) > 0 and manifestaron.nunique() == 1 and manifestaron.iloc[0] == 0
    filas.append({"chequeo": "p_proveedores_que_manifestaron constante",
                  "n": 1 if constante else int(manifestaron.nunique()), "pct": 100.0 if constante else np.nan,
                  "detalle": f"un solo valor (0); {base['p_proveedores_que_manifestaron'].isna().sum()} nulos"
                             if constante else ""})
    filas.append(_fila_chequeo("pago adelantado: 'No' con valor > 0",
                               base["habilita_pago_adelantado"].eq("No") & (base["valor_de_pago_adelantado"] > 0)))
    filas.append(_fila_chequeo("pago adelantado: 'No Definido' con valor > 0",
                               base["habilita_pago_adelantado"].eq("No Definido") & (base["valor_de_pago_adelantado"] > 0)))
    filas.append(_fila_chequeo("pago adelantado: 'Si' con valor = 0",
                               base["habilita_pago_adelantado"].eq("Si") & (base["valor_de_pago_adelantado"] == 0)))
    return pd.DataFrame(filas)


def resumen_atipicos(df, columnas):
    """Cuantiles y conteo de valores extremos por columna numerica (regla 1,5 IQR)."""
    filas = []
    for c in columnas:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        limite = q3 + 1.5 * iqr
        filas.append({
            "columna": c,
            "n": int(s.size),
            "n_cero": int((s == 0).sum()),
            "pct_cero": 100 * float((s == 0).mean()),
            "n_neg": int((s < 0).sum()),
            "min": s.min(),
            "p1": s.quantile(0.01),
            "p50": s.median(),
            "p99": s.quantile(0.99),
            "max": s.max(),
            "n_iqr_alto": int((s > limite).sum()),
            "pct_iqr_alto": 100 * float((s > limite).mean()),
        })
    return pd.DataFrame(filas).sort_values("pct_iqr_alto", ascending=False).reset_index(drop=True)


def cardinalidad_categoricas(df, columnas):
    """Cardinalidad, categoria dominante y peso de las categorias con menos de 1 %."""
    filas = []
    for c in columnas:
        s = df[c]
        vc = s.value_counts(normalize=True, dropna=True)
        raras = vc[vc < 0.01]
        filas.append({
            "columna": c,
            "n_unicos": int(s.nunique(dropna=True)),
            "nulos": int(s.isna().sum()),
            "top": vc.index[0] if len(vc) else None,
            "top_pct": 100 * float(vc.iloc[0]) if len(vc) else np.nan,
            "n_categorias_lt_1pct": int(len(raras)),
            "pct_filas_lt_1pct": 100 * float(raras.sum()),
        })
    return pd.DataFrame(filas).sort_values("n_unicos", ascending=False).reset_index(drop=True)


def graficar_cardinalidad(tabla, top=15, titulo="Cardinalidad de las categoricas"):
    """Barras horizontales con el numero de categorias (escala logaritmica)."""
    t = tabla.head(top).sort_values("n_unicos")
    fig, ax = plt.subplots(figsize=(8, max(2.2, 0.32 * len(t) + 1)))
    ax.barh(t["columna"], t["n_unicos"], color="#55a868")
    for y, (_, r) in enumerate(t.iterrows()):
        ax.text(r["n_unicos"], y, f" {r['n_unicos']:,}", va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("numero de categorias (log)")
    ax.set_title(titulo)
    plt.tight_layout()
    return fig, ax


def graficar_barras_categoricas(df, columnas, top=10, ncols=2, titulo="Composicion de las categoricas"):
    """Panel de barras horizontales con las categorias mas frecuentes de cada columna."""
    nrows = int(np.ceil(len(columnas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, columnas):
        vc = df[c].value_counts(dropna=False).head(top)[::-1]
        etiquetas = [str(v)[:38] for v in vc.index]
        ax.barh(etiquetas, vc.values, color="#4c72b0")
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("contratos")
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes[len(columnas):]:
        ax.set_axis_off()
    fig.suptitle(titulo, fontsize=10)
    plt.tight_layout()
    return fig, axes


# ------------------------------------------------------------------ distribuciones y deriva
def distribucion_numericas(df, columnas):
    """Cuantiles y proporcion de ceros por columna numerica."""
    filas = []
    for c in columnas:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        filas.append({
            "columna": c,
            "n": int(s.size),
            "media": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p1": s.quantile(0.01),
            "p5": s.quantile(0.05),
            "p50": s.median(),
            "p95": s.quantile(0.95),
            "p99": s.quantile(0.99),
            "max": s.max(),
            "pct_cero": 100 * float((s == 0).mean()),
        })
    return pd.DataFrame(filas).reset_index(drop=True)


def graficar_histogramas(df, columnas, ncols=4, escala_log=(), recorte=(0.01, 0.99), titulo="Distribucion de las numericas"):
    """Histogramas con recorte p1-p99 para que los outliers no oculten la forma."""
    nrows = int(np.ceil(len(columnas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, columnas):
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        log = c in escala_log
        if log:
            s = np.log1p(s.clip(lower=0))
        lo, hi = s.quantile(recorte)
        s = s.clip(lo, hi)
        ax.hist(s, bins=30, color="#4c72b0", edgecolor="white", linewidth=0.3)
        etiqueta = f"{c} (log1p, p1-p99)" if log else f"{c} (p1-p99)"
        ax.set_title(etiqueta, fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[len(columnas):]:
        ax.set_axis_off()
    fig.suptitle(titulo, fontsize=10)
    plt.tight_layout()
    return fig, axes


def evolucion_numericas(df, columnas, col_anio="anio_firma"):
    """Mediana de cada feature por anio."""
    return df.groupby(col_anio)[columnas].median()


def deriva_numericas(df, columnas, col_anio="anio_firma"):
    """Medianas por anio, z-score de esas medianas y tabla de cambios.

    Devuelve (medianas, z, cambios). `z` sirve para el heatmap de deriva; `cambios` ordena las
    features por el maximo z absoluto.
    """
    medianas = df.groupby(col_anio)[columnas].median()
    z = medianas.copy().astype(float)
    for c in columnas:
        s = medianas[c]
        sd = s.std(ddof=0)
        z[c] = 0.0 if (pd.isna(sd) or sd == 0) else (s - s.mean()) / sd
    anios = list(medianas.index)
    primero, ultimo = anios[0], anios[-1]
    cambios = pd.DataFrame({
        "mediana_inicio": medianas.loc[primero],
        "mediana_fin": medianas.loc[ultimo],
        "cambio_abs": medianas.loc[ultimo] - medianas.loc[primero],
        "z_max": z.abs().max(),
    })
    if 2023 in medianas.index and 2024 in medianas.index:
        cambios["cambio_2023_2024"] = medianas.loc[2024] - medianas.loc[2023]
    return medianas, z, cambios.sort_values("z_max", ascending=False)


def graficar_deriva_numericas(z, titulo="Deriva numerica: mediana estandarizada por anio"):
    """Heatmap features x anio con la mediana estandarizada (z)."""
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(z.columns) + 1.5)))
    datos = z.T
    im = ax.imshow(datos.values, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_xticks(range(datos.shape[1]))
    ax.set_xticklabels([str(a) for a in datos.columns])
    ax.set_yticks(range(datos.shape[0]))
    ax.set_yticklabels(datos.index, fontsize=7)
    ax.set_title(titulo)
    fig.colorbar(im, ax=ax, label="z de la mediana")
    plt.tight_layout()
    return fig, ax


def resumen_deriva_categoricas(df, columnas, col_anio="anio_firma"):
    """Cambio de composicion entre el primer y el ultimo anio para cada categorica."""
    filas = []
    for c in columnas:
        tabla = pd.crosstab(df[col_anio], df[c], normalize="index")
        if tabla.empty:
            continue
        primero, ultimo = tabla.index[0], tabla.index[-1]
        dif = (tabla.loc[ultimo] - tabla.loc[primero]).abs()
        filas.append({
            "columna": c,
            "top_inicio": tabla.loc[primero].idxmax(),
            "pct_inicio": 100 * tabla.loc[primero].max(),
            "top_fin": tabla.loc[ultimo].idxmax(),
            "pct_fin": 100 * tabla.loc[ultimo].max(),
            "cambio_max_pp": 100 * dif.max(),
            "categoria_cambio": dif.idxmax(),
        })
    return pd.DataFrame(filas).sort_values("cambio_max_pp", ascending=False).reset_index(drop=True)


def graficar_evolucion_categorias(df, columnas, col_anio="anio_firma", top=5, ncols=2,
                                  titulo="Evolucion de la composicion por anio"):
    """Lineas con la participacion de las categorias mas frecuentes, por anio."""
    nrows = int(np.ceil(len(columnas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, columnas):
        tabla = pd.crosstab(df[col_anio], df[c], normalize="index") * 100
        orden = df[c].value_counts().head(top).index
        for cat in orden:
            if cat in tabla.columns:
                ax.plot(tabla.index, tabla[cat], marker="o", ms=3, label=str(cat)[:30])
        ax.set_title(c, fontsize=9)
        ax.set_ylabel("% del anio")
        ax.set_xlabel("")
        ax.legend(fontsize=6, loc="best")
    for ax in axes[len(columnas):]:
        ax.set_axis_off()
    fig.suptitle(titulo, fontsize=10)
    plt.tight_layout()
    return fig, axes


def matriz_correlacion(df, columnas, metodo="spearman"):
    """Matriz de correlacion; descarta columnas constantes o totalmente nulas."""
    datos = df[columnas].apply(pd.to_numeric, errors="coerce")
    datos = datos.loc[:, datos.nunique(dropna=True) > 1]
    return datos.corr(method=metodo)


def pares_correlacionados(corr, umbral=0.7):
    """Pares de features con |r| >= umbral, ordenados de mayor a menor."""
    filas = []
    columnas = list(corr.columns)
    for i, a in enumerate(columnas):
        for b in columnas[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= umbral:
                filas.append({"feature_1": a, "feature_2": b, "r": r, "abs_r": abs(r)})
    return pd.DataFrame(filas).sort_values("abs_r", ascending=False).reset_index(drop=True)


def graficar_matriz_correlacion(corr, titulo="Correlacion entre features numericas"):
    """Heatmap de la matriz de correlacion con los valores anotados."""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            v = corr.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(v) > 0.6 else "black")
    ax.set_title(titulo)
    fig.colorbar(im, ax=ax, label="r")
    plt.tight_layout()
    return fig, ax


# ------------------------------------------------------------------ duracion y madurez
def resumen_duracion_madurez(base):
    """Tabla por anio y por tramo de duracion planeada. No usa la etiqueta."""
    d = base.assign(dur=base["eval_dur_planeada_dias"])
    por_anio = d.groupby("anio_firma").agg(
        n=("y", "size"),
        dur_mediana=("dur", "median"),
        dur_p95=("dur", lambda s: s.quantile(0.95)),
        pct_sin_duracion=("dur", lambda s: 100 * s.isna().mean()),
        pct_duracion_le0=("dur", lambda s: 100 * (s <= 0).mean()),
        pct_duracion_gt3650=("dur", lambda s: 100 * (s > 3650).mean()),
        pct_madura=("eval_madura", lambda s: 100 * s.mean()),
    ).round(2)

    dur = base["eval_dur_planeada_dias"]
    tramo = pd.cut(dur.where(dur > 0), bins=[0, 30, 90, 180, 365, 3650, np.inf],
                   labels=["<=30", "31-90", "91-180", "181-365", "366-3650", ">3650"],
                   include_lowest=True)
    tramo = tramo.cat.add_categories(["Sin duracion / <=0"]).fillna("Sin duracion / <=0")
    por_tramo = (base.assign(tramo=tramo)
                 .groupby("tramo", observed=False)
                 .agg(n=("y", "size"),
                      pct=("y", lambda s: 100 * len(s) / len(base)),
                      pct_madura=("eval_madura", lambda s: 100 * s.mean()))
                 .round(2))
    orden = ["<=30", "31-90", "91-180", "181-365", "366-3650", ">3650", "Sin duracion / <=0"]
    return por_anio, por_tramo.reindex(orden)


def graficar_duracion_madurez(base, por_anio, por_tramo):
    """Histograma de duracion y barras de madurez por anio y por tramo."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    dur = base["eval_dur_planeada_dias"].dropna()
    axes[0].hist(dur.clip(0, 730), bins=40, color="#4c72b0", edgecolor="white", linewidth=0.3)
    axes[0].axvline(dur.median(), color="#c44e52", linestyle="--", linewidth=1,
                    label=f"mediana {dur.median():.0f} d")
    axes[0].set_title("Duracion planeada (recortada a 730 d)")
    axes[0].set_xlabel("dias")
    axes[0].legend()
    axes[1].bar(por_anio.index.astype(str), por_anio["pct_madura"], color="#55a868")
    axes[1].set_title("% maduro por anio de firma")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=45)
    axes[2].bar(range(len(por_tramo)), por_tramo["pct_madura"], color="#55a868")
    axes[2].set_xticks(range(len(por_tramo)))
    axes[2].set_xticklabels(por_tramo.index, rotation=45, ha="right", fontsize=7)
    axes[2].set_title("% maduro por tramo de duracion")
    axes[2].set_ylim(0, 100)
    plt.tight_layout()
    return fig, axes


# ------------------------------------------------------------------ proveedor
def resumen_tipodocproveedor(base):
    """Distribucion del tipo de documento del proveedor (solo el tipo, no la identidad)."""
    t = base["tipodocproveedor"].value_counts(dropna=False).rename_axis("tipo").reset_index(name="n")
    t["pct"] = 100 * t["n"] / len(base)
    t["pct_acumulado"] = t["pct"].cumsum()
    return t


def graficar_tipodocproveedor(tabla, titulo="Tipo de documento del proveedor"):
    """Barras horizontales de la distribucion del tipo de documento."""
    t = tabla.sort_values("n")
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.32 * len(t) + 1)))
    ax.barh([str(x)[:40] for x in t["tipo"]], t["pct"], color="#4c72b0")
    for y, (_, r) in enumerate(t.iterrows()):
        ax.text(r["pct"], y, f" {r['pct']:.2f}% ({r['n']:,})", va="center", fontsize=7)
    ax.set_xlabel("% de contratos")
    ax.set_title(titulo)
    ax.set_xlim(0, max(t["pct"].max() * 1.35, 1))
    plt.tight_layout()
    return fig, ax


# ------------------------------------------------------------------ etiqueta (para v1)
def intervalo_wilson(k, n, z=1.96):
    """Intervalo de confianza de Wilson para una proporcion."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / denom
    margen = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centro - margen, centro + margen


def tasa_por_grupo(df, grupo, col_y="y"):
    """Tasa de y por grupo con n, positivos e intervalo de Wilson al 95 %.

    `grupo` puede ser el nombre de una columna o una Series (por ejemplo, un pd.cut).
    """
    if isinstance(grupo, str):
        g = df[grupo]
        nombre = grupo
    else:
        g = grupo
        nombre = getattr(g, "name", "grupo") or "grupo"
    datos = pd.DataFrame({"grupo": g, "y": df[col_y]})
    tabla = (datos.groupby("grupo", dropna=False)["y"]
             .agg(n="size", positivos="sum")
             .reset_index())
    tabla["tasa"] = tabla["positivos"] / tabla["n"]
    ic = tabla.apply(lambda r: intervalo_wilson(r["positivos"], r["n"]), axis=1)
    tabla["ic_inf"] = [x[0] for x in ic]
    tabla["ic_sup"] = [x[1] for x in ic]
    tabla = tabla.rename(columns={"grupo": nombre})
    return tabla


def graficar_tasa_por_grupo(tabla, col_grupo=None, titulo="Tasa de y por grupo", orden=None,
                            horizontal=False, ax=None):
    """Barras con la tasa de y y su intervalo de Wilson."""
    if col_grupo is None:
        col_grupo = tabla.columns[0]
    t = tabla.copy()
    if orden is not None:
        t[col_grupo] = pd.Categorical(t[col_grupo], categories=orden, ordered=True)
        t = t.sort_values(col_grupo)
    t = t.reset_index(drop=True)
    y = 100 * t["tasa"].to_numpy()
    err = np.vstack([100 * (t["tasa"] - t["ic_inf"]).to_numpy(),
                     100 * (t["ic_sup"] - t["tasa"]).to_numpy()])
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.8))
    else:
        fig = ax.figure
    x = np.arange(len(t))
    if horizontal:
        ax.barh(x, y, xerr=err, color="#4c72b0", capsize=3)
        ax.set_yticks(x)
        ax.set_yticklabels([str(v) for v in t[col_grupo]], fontsize=7)
        ax.set_xlabel("tasa de y (%)")
    else:
        ax.bar(x, y, yerr=err, color="#4c72b0", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in t[col_grupo]], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("tasa de y (%)")
    for i, r in t.iterrows():
        if horizontal:
            ax.text(r["tasa"] * 100 + 0.5, i, f"n={r['n']:,}", va="center", fontsize=6)
        else:
            ax.text(i, r["ic_sup"] * 100 + 0.4, f"n={r['n']:,}", ha="center", fontsize=6)
    ax.set_title(titulo)
    ax.set_ylim(0, max(100 * t["ic_sup"].max() * 1.2, 5))
    plt.tight_layout()
    return fig, ax


def tasa_por_anio(df, col_y="y", col_anio="anio_firma"):
    """Tasa de y por anio con intervalo de Wilson."""
    return tasa_por_grupo(df, col_anio, col_y)


def graficar_tasa_por_anio(tabla, col_anio="anio_firma", titulo="Tasa de y por anio"):
    """Linea con intervalo de Wilson por anio."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = 100 * tabla["tasa"].to_numpy()
    err = np.vstack([100 * (tabla["tasa"] - tabla["ic_inf"]).to_numpy(),
                     100 * (tabla["ic_sup"] - tabla["tasa"]).to_numpy()])
    ax.errorbar(tabla[col_anio], y, yerr=err, fmt="o-", capsize=3, color="#4c72b0")
    for _, r in tabla.iterrows():
        ax.text(r[col_anio], 100 * r["ic_sup"] + 0.5, f"{100*r['tasa']:.1f}%\nn={r['n']:,}",
                ha="center", fontsize=6)
    ax.set_xlabel("anio de firma")
    ax.set_ylabel("tasa de y (%)")
    ax.set_title(titulo)
    plt.tight_layout()
    return fig, ax


def tasa_entidad_historica(df, col_entidad="nit_entidad", col_y="y", col_anio="anio_firma",
                           anio_objetivo=2024, n_grupos=5):
    """Tasa historica de la entidad con anios anteriores y tasa observada en el anio objetivo.

    Evita fuga temporal: para cada contrato del anio objetivo usa solo la tasa de su entidad en
    anios < anio_objetivo. Las entidades sin historia van a un grupo aparte.
    """
    hist = (df[df[col_anio] < anio_objetivo]
            .groupby(col_entidad)[col_y]
            .agg(positivos="sum", n="size"))
    hist["tasa"] = hist["positivos"] / hist["n"]
    try:
        hist["grupo"] = pd.qcut(hist["tasa"], q=n_grupos, duplicates="drop")
    except ValueError:
        hist["grupo"] = "Unico"
    objetivo = df[df[col_anio] == anio_objetivo].copy()
    objetivo["grupo_hist"] = objetivo[col_entidad].map(hist["grupo"])
    objetivo["grupo_hist"] = objetivo["grupo_hist"].astype("object").fillna("Sin historia")
    tabla = tasa_por_grupo(objetivo, "grupo_hist", col_y)
    return tabla, hist
