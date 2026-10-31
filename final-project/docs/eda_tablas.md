# Tablas EDA — base v1_A (corte 26-oct-2026)

Documento de trabajo para cerrar las interpretaciones pendientes de `notebooks/_borradores/03_eda.ipynb` (§4.3, §5, §6.1–§6.7, §7.2 y §8). No edita el notebook. Todas las tablas se generaron en esta sesión con la base v1_A y el código que aparece en cada bloque `<details>`.

## §0. Base, subconjuntos y convenciones

La base v1_A tiene **452.494 contratos** firmados entre 2019 y 2025 (muestra del 10 % por dígito 4 del `id_contrato`, autoponderada). La tasa global de `y` es **27,635 %** (27,6 %).

### Tabla 0.1. Subconjuntos usados

**Qué muestra:** Tamaño y tasa de y de cada subconjunto que se usa en el documento.

| subconjunto | n | % y=1 | uso |
|---|---|---|---|
| Todos 2019–2025 | 452.494 | 27,6 % | §4.3, §5, §6.1–§6.5, §6.7, §7.2, §8 |
| Train 2019–2023 | 261.831 | 28,8 % | Referencia de medianas de H5 e historia de H6; no hay tabla de y de todo el train |
| Test 2024 | 88.546 | 24,7 % | §6.6 |
| Oct-2021–dic-2023 | 166.590 | 29,2 % | Calendario de §6.4 |

**Nota:** el calendario no es train ni test; es un periodo pedido para revisar el calendario de contratos. Cada sección repite el subconjunto que usa.

<details>
<summary>Código que genera la tabla</summary>

```python
train = base[base["anio_firma"] <= 2023]
test = base[base["anio_firma"] == 2024]
periodo = Xy.loc[
    (Xy["fecha_de_firma"] >= "2021-10-01")
    & (Xy["fecha_de_firma"] < "2024-01-01")
]
for nombre, df in [
    ("Todos 2019–2025", base),
    ("Train 2019–2023", train),
    ("Test 2024", test),
    ("Oct-2021–dic-2023", periodo),
]:
    print(nombre, len(df), 100 * df["y"].mean())
```

</details>

### Convenciones

- **Columnas estándar:** `categoría | n | % del total | % y=1 | diferencia (pp)`. `% del total` usa el n del subconjunto; `diferencia` es `% y=1` menos la tasa base del mismo subconjunto.
- **Categorías con n < 500:** se agrupan en `Otras`, indicando cuántas categorías y cuántos contratos entraron. En variables numéricas se conservan los cortes del notebook y se reporta el n de cada tramo; no se aplica la agrupación n < 500 a los tramos.
- **Orden:** las categóricas no ordenadas se ordenan por `% y=1`; los tramos numéricos, los meses, los grupos de madurez y los grupos históricos conservan su orden natural.
- **Intervalos:** las tablas especiales (madurez por año, etiquetas por año, H6) incluyen IC 95 % de Wilson; las tablas estándar no los repiten.
- **AUC univariado:** en numéricas se usa el valor observado; en categóricas, la tasa de y por categoría calculada en la misma muestra. Es descriptivo, no una validación.
- **`NO EJECUTADO`:** se escribe cuando la columna no está en `data/raw/f2` o no tiene un valor numérico. No se estima.
- Los snippets de cada tabla suponen la carga y los helpers de §0.

### Carga de la base

<details>
<summary>Carga de la base v1_A</summary>

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAIZ = Path(r"C:\Users\kevineafit\OneDrive\Escritorio\SEM2026-2\ML\workshops\repositorio\aa-eda\final-project")
BASE = RAIZ / "data" / "processed" / "v1_A" / "base_contratos.parquet"
sys.path.insert(0, str(RAIZ))

from src import eda
from src import features as F

base, meta = eda.cargar_base(BASE)
X = F.construir_features(base)
Xy = X.assign(
    y=base["y"],
    anio_firma=base["anio_firma"],
    fecha_de_firma=pd.to_datetime(base["fecha_de_firma"]),
)
```

</details>

### Helpers de tablas y formato

<details>
<summary>Funciones auxiliares usadas por los snippets</summary>

```python
import textwrap

def esc(x):
    return str(x).replace("|", r"\|")


def fmt_int(x):
    return f"{int(x):,}".replace(",", ".")


def fmt_pct(x, dec=1):
    if pd.isna(x):
        return "—"
    if abs(x) < 0.5 * 10 ** (-dec):
        x = 0.0
    return f"{x:.{dec}f}".replace(".", ",") + " %"


def fmt_pp(x, dec=1):
    if pd.isna(x):
        return "—"
    if abs(x) < 0.5 * 10 ** (-dec):
        x = 0.0
    return f"{x:+.{dec}f}".replace(".", ",") + " pp"


def fmt_auc(x):
    if x is None or pd.isna(x):
        return "NO EJECUTADO"
    dec = 4 if abs(x) > 0.999 else 3
    return f"{x:.{dec}f}".replace(".", ",")


def md(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def details(code, summary="Código que genera la tabla"):
    code = textwrap.dedent(code).strip()
    return (
        f"<details>\n<summary>{summary}</summary>\n\n"
        f"```python\n{code}\n```\n\n</details>"
    )


def bloque(muestra, md_text, nota="", snippet="", summary="Código que genera la tabla"):
    partes = [f"**Qué muestra:** {muestra}", "", md_text]
    if nota:
        partes += ["", nota]
    if snippet:
        partes += ["", details(snippet, summary)]
    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Tablas estándar
# ---------------------------------------------------------------------------
def tabla_estandar(df, col, y="y", top=None, min_n=500, agrupar_pequenas=True,
                   etiqueta_otros="Otras", orden="tasa", orden_natural=None,
                   nombre=None):
    """Tabla categoría | n | % del total | % y=1 | diferencia pp vs base."""
    s0 = df[col] if isinstance(col, str) else col.copy()
    s0 = pd.Series(s0, index=df.index).astype("object")
    s0 = s0.where(s0.notna(), "Sin dato")
    nombre = nombre or (col if isinstance(col, str) else getattr(col, "name", "categoria"))

    counts0 = s0.value_counts(dropna=False)
    keep = set(counts0.head(top).index) if top is not None else set(counts0.index)
    small = {c for c, n in counts0.items() if n < min_n} if agrupar_pequenas else set()
    agrupadas = (set(counts0.index) - keep) | small
    s = s0.where(~s0.isin(agrupadas), etiqueta_otros)
    s.name = nombre

    d = df.assign(_grupo=s)
    g = (d.groupby("_grupo", dropna=False)[y]
         .agg(n="size", positivos="sum")
         .reset_index()
         .rename(columns={"_grupo": nombre}))
    n_total = len(d)
    base_rate = 100 * d[y].mean()
    g["pct_total"] = 100 * g["n"] / n_total
    g["tasa"] = 100 * g["positivos"] / g["n"]
    g["dif"] = g["tasa"] - base_rate

    if orden == "tasa":
        g = g.sort_values(["tasa", "n"], ascending=[False, False]).reset_index(drop=True)
    elif orden == "natural":
        cats = orden_natural if orden_natural is not None else list(g[nombre])
        g[nombre] = pd.Categorical(g[nombre], categories=cats, ordered=True)
        g = g.sort_values(nombre).reset_index(drop=True)

    rows = []
    for r in g.to_dict("records"):
        rows.append([
            esc(r[nombre]), fmt_int(r["n"]), fmt_pct(r["pct_total"]),
            fmt_pct(r["tasa"]), fmt_pp(r["dif"]),
        ])
    rows.append(["Total", fmt_int(n_total), "100,0 %", fmt_pct(base_rate), "0,0 pp"])
    md_text = md([nombre, "n", "% del total", "% y=1", "diferencia (pp)"], rows)

    info = {
        "n_total": n_total,
        "base_rate": base_rate,
        "n_categorias": len(counts0),
        "n_agrupadas": len(agrupadas),
        "n_filas_agrupadas": int(s0.isin(agrupadas).sum()),
        "agrupadas": sorted([str(x) for x in agrupadas]),
        "tabla": g,
    }
    return md_text, info


def tabla_tramos(df, col, bins, labels, y="y", etiqueta_nulos="Sin dato", nombre=None):
    s0 = df[col] if isinstance(col, str) else col
    s = pd.cut(s0, bins=bins, labels=labels, right=True, include_lowest=True)
    if etiqueta_nulos is not None:
        if etiqueta_nulos not in s.cat.categories:
            s = s.cat.add_categories([etiqueta_nulos])
        s = s.fillna(etiqueta_nulos)
    s = s.astype("object")
    nombre = nombre or (col if isinstance(col, str) else getattr(col, "name", "tramo"))
    s.name = nombre
    cats = list(labels) + ([etiqueta_nulos] if etiqueta_nulos is not None else [])
    return tabla_estandar(
        df.assign(**{nombre: s}), nombre, y=y, min_n=0, agrupar_pequenas=False,
        orden="natural", orden_natural=cats,
    )


# ---------------------------------------------------------------------------
# AUC univariado
# ---------------------------------------------------------------------------
def auc_num(df, col, y="y"):
    x = df[[col, y]].dropna()
    if len(x) == 0 or x[col].nunique() <= 1:
        return None, len(x)
    return float(roc_auc_score(x[y], x[col])), len(x)


def auc_cat(df, col, y="y"):
    x = df[[col, y]].dropna()
    if len(x) == 0 or x[col].nunique() <= 1:
        return None, len(x)
    tasas = x.groupby(col)[y].mean()
    return float(roc_auc_score(x[y], x[col].map(tasas))), len(x)
```

</details>

## §4.3 Madurez y etiqueta

**Subconjunto:** todos los contratos 2019–2025 (n = 452.494; y = 27,6 %). No usa test 2024.

### Tabla 4.3.1. Fracción madura por año

**Qué muestra:** Fracción de contratos con `eval_madura = True` por año de firma, con intervalo de Wilson del 95 %.

| año | n | n maduros | n no maduros | % maduro | IC 95 % |
|---|---|---|---|---|---|
| 2019 | 14.407 | 14.406 | 1 | 99,99 % | 99,96 % – 100,00 % |
| 2020 | 36.107 | 36.076 | 31 | 99,91 % | 99,88 % – 99,94 % |
| 2021 | 56.469 | 56.370 | 99 | 99,82 % | 99,79 % – 99,86 % |
| 2022 | 71.091 | 70.899 | 192 | 99,73 % | 99,69 % – 99,77 % |
| 2023 | 83.757 | 83.498 | 259 | 99,69 % | 99,65 % – 99,73 % |
| 2024 | 88.546 | 88.088 | 458 | 99,48 % | 99,43 % – 99,53 % |
| 2025 | 102.117 | 99.702 | 2.415 | 97,64 % | 97,54 % – 97,73 % |
| Total | 452.494 | 449.039 | 3.455 | 99,24 % | 99,21 % – 99,26 % |

**Nota:** `eval_madura = True` si fin planeado + 60 días ≤ fecha de corte (26-oct-2026).

<details>
<summary>Código que genera la tabla</summary>

```python
madurez_anio = (
    base.groupby("anio_firma")["eval_madura"]
    .agg(n="size", maduros="sum")
    .assign(pct_madura=lambda d: 100 * d["maduros"] / d["n"])
)
madurez_anio[["ic_inf_pct", "ic_sup_pct"]] = [
    [100 * x for x in eda.intervalo_wilson(r.maduros, r.n)]
    for r in madurez_anio.itertuples()
]
```

</details>

### Tabla 4.3.2. Tasa de y por madurez

**Qué muestra:** Tasa de y en contratos maduros y no maduros, con n, % del total y diferencia en pp frente a la tasa base.

| madurez | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| Maduro | 449.039 | 99,2 % | 27,7 % | +0,0 pp |
| No maduro | 3.455 | 0,8 % | 22,5 % | -5,2 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** las dos categorías tienen n ≥ 500; no se agrupó ninguna. Tasa base del subconjunto: 27,6 %. De los 3.455 no maduros, 776 ya tienen y=1 y 2.679 tienen y=0.

<details>
<summary>Código que genera la tabla</summary>

```python
base_madurez = base.assign(
    madurez=pd.Categorical(
        base["eval_madura"].map({True: "Maduro", False: "No maduro"}),
        categories=["Maduro", "No maduro"], ordered=True,
    )
)
tabla_estandar(
    base_madurez, "madurez", orden="natural",
    orden_natural=["Maduro", "No maduro"],
)
print(base.groupby("eval_madura")["y"].agg(["size", "sum"]))
```

</details>

### Tabla 4.3.3. Tasa de y por tramo de duración planeada

**Qué muestra:** Tasa de y por tramo de `eval_dur_planeada_dias`, con `Duración 0` y `Sin dato` separados.

| tramo_dur | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| Duración 0 | 24.566 | 5,4 % | 24,7 % | -2,9 pp |
| 1-30 | 42.517 | 9,4 % | 20,5 % | -7,1 pp |
| 31-90 | 108.000 | 23,9 % | 23,1 % | -4,6 pp |
| 91-180 | 144.602 | 32,0 % | 27,9 % | +0,2 pp |
| 181-365 | 126.781 | 28,0 % | 33,7 % | +6,1 pp |
| 366-3650 | 5.104 | 1,1 % | 42,2 % | +14,6 pp |
| >3650 | 125 | 0,0 % | 27,2 % | -0,4 pp |
| Sin dato | 799 | 0,2 % | 15,0 % | -12,6 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** tramos del notebook. `Duración 0` = `p_duracion` registrada como 0 (24.566 contratos; se trata como duración desconocida para `eval_madura`; 23.699 maduros y 867 no maduros, firmados entre octubre y diciembre de 2025); `Sin dato` = 799 sin duración. La regla n < 500 se aplica a categóricas; en tramos numéricos se conservan los cortes y se reporta el n de cada tramo (`>3650` = 125; `Sin dato` = 799). Dentro de 366–3.650 días hay 616 comodatos con 6,7 % de y; sin ellos, el tramo queda en 47,1 %. Entre maduros, ese tramo tiene 3.498 contratos y 57,0 % de y; los 1.606 no maduros del mismo tramo tienen 10,1 %, así que el 42,2 % es el promedio ponderado.

<details>
<summary>Código que genera la tabla</summary>

```python
dur = base["eval_dur_planeada_dias"]
tramo = pd.cut(
    dur.where(dur > 0),
    bins=[0, 30, 90, 180, 365, 3650, np.inf],
    labels=["1-30", "31-90", "91-180", "181-365", "366-3650", ">3650"],
    include_lowest=True,
)
tramo = tramo.cat.add_categories(["Duración 0", "Sin dato"])
tramo = tramo.mask(dur.eq(0), "Duración 0").fillna("Sin dato")
tramo = pd.Categorical(
    tramo,
    categories=["Duración 0", "1-30", "31-90", "91-180", "181-365",
                "366-3650", ">3650", "Sin dato"],
    ordered=True,
)
tabla_estandar(
    base.assign(tramo_dur=tramo), "tramo_dur",
    agrupar_pequenas=False, orden="natural",
    orden_natural=["Duración 0", "1-30", "31-90", "91-180", "181-365",
                   "366-3650", ">3650", "Sin dato"],
)
banda = base[(dur > 365) & (dur <= 3650)]
com = banda[banda["tipo_de_contrato"] == "Comodato"]
resto = banda[banda["tipo_de_contrato"] != "Comodato"]
print("banda 366-3650:", len(banda), 100 * banda["y"].mean())
print("comodato:", len(com), 100 * com["y"].mean())
print("sin comodato:", len(resto), 100 * resto["y"].mean())
print(banda.groupby("eval_madura")["y"].agg(["size", "mean"]))
```

</details>

### Tabla 4.3.4. Madurez y tasa de y por año

**Qué muestra:** Contratos maduros y no maduros por año de firma, con n y tasa de y en cada grupo.

| año | n | n maduros | % y maduros | n no maduros | % y no maduros | diferencia no maduro − maduro (pp) |
|---|---|---|---|---|---|---|
| 2019 | 14.407 | 14.406 | 22,2 % | 1 | 0,0 % | -22,2 pp |
| 2020 | 36.107 | 36.076 | 29,3 % | 31 | 48,4 % | +19,1 pp |
| 2021 | 56.469 | 56.370 | 27,8 % | 99 | 10,1 % | -17,7 pp |
| 2022 | 71.091 | 70.899 | 29,5 % | 192 | 10,4 % | -19,0 pp |
| 2023 | 83.757 | 83.498 | 30,0 % | 259 | 12,0 % | -18,1 pp |
| 2024 | 88.546 | 88.088 | 24,7 % | 458 | 11,6 % | -13,2 pp |
| 2025 | 102.117 | 99.702 | 27,1 % | 2.415 | 26,8 % | -0,4 pp |
| Subtotal 2019–2024 | 350.377 | 349.337 | 27,8 % | 1.040 | 12,4 % | -15,4 pp |
| Total | 452.494 | 449.039 | 27,7 % | 3.455 | 22,5 % | -5,2 pp |

**Nota:** `eval_madura` según fin planeado + 60 días ≤ 26-oct-2026; si la duración es 0 o falta, firma + 365 días. `Diferencia` = % y no maduros − % y maduros. `Subtotal 2019–2024` agrupa los años anteriores a 2025.

<details>
<summary>Código que genera la tabla</summary>

```python
def fila_madurez(df, etiqueta):
    mad = df[df["eval_madura"]]
    no_mad = df[~df["eval_madura"]]
    tasa_mad = 100 * mad["y"].mean()
    tasa_no = 100 * no_mad["y"].mean() if len(no_mad) else np.nan
    return [etiqueta, len(df), len(mad), tasa_mad,
            len(no_mad), tasa_no, tasa_no - tasa_mad]

filas = [fila_madurez(base[base["anio_firma"] == a], str(a))
         for a in sorted(base["anio_firma"].unique())]
filas.append(fila_madurez(
    base[base["anio_firma"] <= 2024], "Subtotal 2019–2024"
))
filas.append(fila_madurez(base, "Total"))
for et, n, nm, tm, nn, tn, dif in filas:
    print(et, n, nm, tm, nn, tn, dif)
```

</details>

### Tabla 4.3.5. No maduros de 2019–2024 por tipo de contrato

**Qué muestra:** Composición de los no maduros anteriores a 2025 por tipo de contrato, con tasa de y y mediana de la duración planeada.

| grupo | n | % de no maduros 2019–2024 | % y=1 | mediana duración (días) |
|---|---|---|---|---|
| Comodato | 305 | 29,3 % | 1,6 % | 1.825 |
| Otro | 398 | 38,3 % | 7,0 % | 1.460 |
| Resto de tipos | 337 | 32,4 % | 28,5 % | 1.825 |
| Total | 1.040 | 100,0 % | 12,4 % | 1.825 |

**Nota:** subconjunto = 1.040 contratos no maduros firmados en 2019–2024. `Resto de tipos` = todos los tipos distintos de Comodato y Otro (principalmente prestación de servicios, 159; arrendamiento de inmuebles, 37). No se agrupan Comodato ni Otro aunque tengan n < 500: son las dos categorías que explican el subtotal. Además, 97 contratos superan los 3.650 días y 6 los 100.000 días (el máximo es 4.078.904.000 días): son errores de captura, no plazos.

<details>
<summary>Código que genera la tabla</summary>

```python
viejo = base[(~base["eval_madura"]) & (base["anio_firma"] <= 2024)].copy()
viejo["grupo"] = viejo["tipo_de_contrato"].where(
    viejo["tipo_de_contrato"].isin(["Comodato", "Otro"]), "Resto de tipos"
)
res = (
    viejo.groupby("grupo")
    .agg(n=("y", "size"), tasa=("y", "mean"),
         mediana=("eval_dur_planeada_dias", "median"))
    .reindex(["Comodato", "Otro", "Resto de tipos"])
)
for g, r in res.iterrows():
    print(g, r["n"], 100 * r["n"] / len(viejo),
          100 * r["tasa"], r["mediana"])
print("Total", len(viejo), 100 * viejo["y"].mean(),
      viejo["eval_dur_planeada_dias"].median())
print("duración > 3650:", int((viejo["eval_dur_planeada_dias"] > 3650).sum()))
print("duración > 100000:", int((viejo["eval_dur_planeada_dias"] > 100000).sum()))
print("máximo:", viejo["eval_dur_planeada_dias"].max())
```

</details>

## §5 Etiqueta `y` frente a `y_dias` y `y_modif`

**Subconjunto:** todos los contratos 2019–2025 (n = 452.494; y = 27,6 %). No usa test 2024.

### Tabla 5.1. Tasa de cada etiqueta por año

**Qué muestra:** Tasa de `y`, `y_dias` y `y_modif` por año de firma, con IC 95 % de Wilson para `y` y diferencias en pp.

| año | n | % y=1 | IC 95 % y | % y_dias=1 | % y_modif=1 | diferencia y − y_dias (pp) | diferencia y − y_modif (pp) |
|---|---|---|---|---|---|---|---|
| 2019 | 14.407 | 22,2 % | 21,5 % – 22,8 % | 11,3 % | 22,1 % | +10,9 pp | +0,0 pp |
| 2020 | 36.107 | 29,3 % | 28,9 % – 29,8 % | 13,0 % | 29,3 % | +16,3 pp | +0,0 pp |
| 2021 | 56.469 | 27,8 % | 27,4 % – 28,2 % | 9,8 % | 27,8 % | +18,0 pp | +0,0 pp |
| 2022 | 71.091 | 29,4 % | 29,1 % – 29,7 % | 9,4 % | 29,4 % | +20,0 pp | +0,0 pp |
| 2023 | 83.757 | 30,0 % | 29,7 % – 30,3 % | 12,6 % | 30,0 % | +17,4 pp | +0,0 pp |
| 2024 | 88.546 | 24,7 % | 24,4 % – 24,9 % | 12,1 % | 24,6 % | +12,6 pp | +0,0 pp |
| 2025 | 102.117 | 27,1 % | 26,9 % – 27,4 % | 20,0 % | 27,0 % | +7,1 pp | +0,1 pp |
| Total | 452.494 | 27,6 % | 27,5 % – 27,8 % | 13,3 % | 27,6 % | +14,3 pp | +0,0 pp |

**Nota:** `y_modif` no tiene faltantes; `y` y `y_dias` tampoco. Las diferencias se calculan sobre puntos porcentuales.

<details>
<summary>Código que genera la tabla</summary>

```python
comp = eda.comparar_etiquetas(base, etiquetas=("y", "y_dias", "y_modif"))
tasas = comp.pivot(index="anio_firma", columns="etiqueta", values="tasa") * 100
inf = comp.pivot(index="anio_firma", columns="etiqueta", values="ic_inf") * 100
sup = comp.pivot(index="anio_firma", columns="etiqueta", values="ic_sup") * 100
n_anio = comp.groupby("anio_firma")["n"].first()
```

</details>

### Tabla 5.2. Discrepancias entre `y` y `y_dias` por año

**Qué muestra:** Conteo de acuerdos y discrepancias entre `y` y `y_dias` por año, y porcentaje de discrepancias.

| año | n | y=1, y_dias=0 | y=0, y_dias=1 | iguales | % discrepancias |
|---|---|---|---|---|---|
| 2019 | 14.407 | 1.569 | 0 | 12.838 | 10,9 % |
| 2020 | 36.107 | 5.884 | 0 | 30.223 | 16,3 % |
| 2021 | 56.469 | 10.170 | 0 | 46.299 | 18,0 % |
| 2022 | 71.091 | 14.194 | 0 | 56.897 | 20,0 % |
| 2023 | 83.757 | 14.565 | 0 | 69.192 | 17,4 % |
| 2024 | 88.546 | 11.136 | 0 | 77.410 | 12,6 % |
| 2025 | 102.117 | 7.241 | 0 | 94.876 | 7,1 % |
| Total | 452.494 | 64.759 | 0 | 387.735 | 14,3 % |

**Nota:** `y=1, y_dias=0` son contratos positivos solo por Modificaciones; `y=0, y_dias=1` no aparece en ningún año.

<details>
<summary>Código que genera la tabla</summary>

```python
disc = eda.discrepancias_etiquetas(base, col_a="y", col_b="y_dias")
for r in disc.itertuples():
    dif = r.a_1_b_0 + r.a_0_b_1
    print(r.anio_firma, r.n, r.a_1_b_0, r.a_0_b_1, r.iguales, 100 * dif / r.n)
```

</details>

### Tabla 5.3. Tabla 2×2 de `y` y `y_modif`

**Qué muestra:** Conteo de contratos en las cuatro combinaciones de `y` y `y_modif`, y porcentaje del total.

| y | y_modif | n | % del total |
|---|---|---|---|
| 0 | 0 | 327.446 | 72,4 % |
| 0 | 1 | 0 | 0,0 % |
| 1 | 0 | 150 | 0,0 % |
| 1 | 1 | 124.898 | 27,6 % |
| Total | — | 452.494 | 100,0 % |

**Nota:** no hay contratos con `y=0` y `y_modif=1`; los 150 contratos con `y=1` y `y_modif=0` son la única discrepancia.

<details>
<summary>Código que genera la tabla</summary>

```python
ct = pd.crosstab(base["y"], base["y_modif"])
for a in [0, 1]:
    for b in [0, 1]:
        k = int(ct.loc[a, b]) if (a in ct.index and b in ct.columns) else 0
        print(a, b, k, 100 * k / len(base))
```

</details>

### Tabla 5.4. Distribución de `y_dias` dentro de cada valor de `y`

**Qué muestra:** Conteo de `y_dias` dentro de `y=0` y `y=1`, y porcentaje de `y_dias=1` en cada fila.

| y | n | y_dias=0 | y_dias=1 | % y_dias=1 |
|---|---|---|---|---|
| 0 | 327.446 | 327.446 | 0 | 0,0 % |
| 1 | 125.048 | 64.759 | 60.289 | 48,2 % |
| Total | 452.494 | 392.205 | 60.289 | 13,3 % |

**Nota:** `y_dias=1` implica `y=1`; no hay contratos con `y=0` y `y_dias=1`.

<details>
<summary>Código que genera la tabla</summary>

```python
ct2 = pd.crosstab(base["y"], base["y_dias"])
for a in [0, 1]:
    k0, k1 = int(ct2.loc[a, 0]), int(ct2.loc[a, 1])
    print(a, k0 + k1, k0, k1, 100 * k1 / (k0 + k1))
```

</details>

## §6.1 H1. Los procesos apresurados terminan más con el plazo ampliado

**Subconjunto:** todos los contratos 2019–2025 (n = 452.494; y = 27,6 %). No usa test 2024.

### Tabla 6.1.1. Tasa de y por días entre publicación y firma

**Qué muestra:** Tasa de y por tramo de `dias_publicacion_a_firma` (días entre la publicación del proceso y la firma).

| dias_publicacion_a_firma | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| 0 | 165.613 | 36,6 % | 25,5 % | -2,1 pp |
| 1-7 | 216.256 | 47,8 % | 28,3 % | +0,7 pp |
| 8-15 | 33.810 | 7,5 % | 27,8 % | +0,2 pp |
| 16-30 | 21.228 | 4,7 % | 31,2 % | +3,6 pp |
| 31-60 | 9.738 | 2,2 % | 40,0 % | +12,4 pp |
| 61-120 | 3.259 | 0,7 % | 36,6 % | +9,0 pp |
| >120 | 2.210 | 0,5 % | 14,8 % | -12,8 pp |
| Sin dato | 380 | 0,1 % | 20,0 % | -7,6 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** cortes del notebook. `0` agrupa los días ≤ 0. `Sin dato` = 380 contratos sin `p_fecha_de_publicacion_del`; el texto del notebook dice 229, pero la tabla ejecutada da 380. No se aplica la regla n < 500 a tramos numéricos.

<details>
<summary>Código que genera la tabla</summary>

```python
tabla_tramos(
    Xy, "dias_publicacion_a_firma",
    bins=[-np.inf, 0, 7, 15, 30, 60, 120, np.inf],
    labels=["0", "1-7", "8-15", "16-30", "31-60", "61-120", ">120"],
)
```

</details>

### Tabla 6.1.1.b. Tasa de y por días entre publicación y firma (train 2019–2023)

**Qué muestra:** Tasa de y por tramo de `dias_publicacion_a_firma` (días entre la publicación del proceso y la firma).

**Subconjunto:** train 2019–2023 (n = 261.831; y = 28,8 %).

| dias_publicacion_a_firma | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| 0 | 91.896 | 35,1 % | 26,6 % | -2,3 pp |
| 1-7 | 124.781 | 47,7 % | 29,4 % | +0,6 pp |
| 8-15 | 20.628 | 7,9 % | 30,4 % | +1,5 pp |
| 16-30 | 13.541 | 5,2 % | 31,4 % | +2,6 pp |
| 31-60 | 6.772 | 2,6 % | 40,1 % | +11,3 pp |
| 61-120 | 2.449 | 0,9 % | 35,7 % | +6,9 pp |
| >120 | 1.753 | 0,7 % | 13,9 % | -15,0 pp |
| Sin dato | 11 | 0,0 % | 18,2 % | -10,7 pp |
| Total | 261.831 | 100,0 % | 28,8 % | 0,0 pp |

**Nota:** Mismos cortes del notebook. `Sin dato` = contratos sin `p_fecha_de_publicacion_del`. No se aplica la regla n < 500 a tramos numéricos.

**Dirección frente a 2019–2025:** Se mantiene: el máximo sigue en 31–60 días (40,1 % vs 40,0 %) y el tramo 0 sigue por debajo de la base (26,6 % vs 25,5 %).

<details>
<summary>Código que genera la tabla</summary>

```python
train = Xy[Xy["anio_firma"] <= 2023]
tabla_tramos(
    train, "dias_publicacion_a_firma",
    bins=[-np.inf, 0, 7, 15, 30, 60, 120, np.inf],
    labels=["0", "1-7", "8-15", "16-30", "31-60", "61-120", ">120"],
)
```

</details>

## §6.2 H2. Poca competencia se asocia con más ampliaciones del plazo

**Subconjunto:** todos los contratos 2019–2025 (n = 452.494; y = 27,6 %). No usa test 2024.

### Tabla 6.2.1. Tasa de y por respuestas al procedimiento

**Qué muestra:** Tasa de y por tramo de `p_respuestas_al_procedimiento`.

| p_respuestas_al_procedimiento | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| 0 | 408.894 | 90,4 % | 27,5 % | -0,1 pp |
| 1 | 18.009 | 4,0 % | 25,2 % | -2,5 pp |
| 2-3 | 11.449 | 2,5 % | 25,5 % | -2,1 pp |
| 4-9 | 9.530 | 2,1 % | 30,4 % | +2,8 pp |
| 10+ | 4.263 | 0,9 % | 49,0 % | +21,3 pp |
| Sin dato | 349 | 0,1 % | 19,5 % | -8,2 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** cortes del notebook. `Sin dato` = 349 contratos sin fila de Procesos. No se aplica la regla n < 500 a tramos numéricos.

<details>
<summary>Código que genera la tabla</summary>

```python
tabla_tramos(
    Xy, "p_respuestas_al_procedimiento",
    bins=[-np.inf, 0, 1, 3, 9, np.inf],
    labels=["0", "1", "2-3", "4-9", "10+"],
)
```

</details>

### Tabla 6.2.1.b. Tasa de y por respuestas al procedimiento (train 2019–2023)

**Qué muestra:** Tasa de y por tramo de `p_respuestas_al_procedimiento`.

**Subconjunto:** train 2019–2023 (n = 261.831; y = 28,8 %).

| p_respuestas_al_procedimiento | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| 0 | 235.700 | 90,0 % | 28,9 % | +0,0 pp |
| 1 | 10.079 | 3,8 % | 24,7 % | -4,1 pp |
| 2-3 | 7.220 | 2,8 % | 24,7 % | -4,2 pp |
| 4-9 | 6.163 | 2,4 % | 30,1 % | +1,2 pp |
| 10+ | 2.663 | 1,0 % | 50,2 % | +21,4 pp |
| Sin dato | 6 | 0,0 % | 16,7 % | -12,2 pp |
| Total | 261.831 | 100,0 % | 28,8 % | 0,0 pp |

**Nota:** Mismos cortes del notebook. `Sin dato` = contratos sin fila de Procesos. No se aplica la regla n < 500 a tramos numéricos.

**Dirección frente a 2019–2025:** Se mantiene: 0 respuestas queda en la base (28,9 % vs 27,5 %) y 10+ sigue muy por encima (50,2 % vs 49,0 %).

<details>
<summary>Código que genera la tabla</summary>

```python
train = Xy[Xy["anio_firma"] <= 2023]
tabla_tramos(
    train, "p_respuestas_al_procedimiento",
    bins=[-np.inf, 0, 1, 3, 9, np.inf],
    labels=["0", "1", "2-3", "4-9", "10+"],
)
```

</details>

### Tabla 6.2.2. Tasa de y por proveedores invitados

**Qué muestra:** Tasa de y por tramo de `p_proveedores_invitados`.

| p_proveedores_invitados | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| 0 | 442.688 | 97,8 % | 27,5 % | -0,2 pp |
| 1-4 | 4.996 | 1,1 % | 32,1 % | +4,4 pp |
| 5-9 | 1.079 | 0,2 % | 41,0 % | +13,3 pp |
| 10+ | 3.382 | 0,7 % | 41,7 % | +14,0 pp |
| Sin dato | 349 | 0,1 % | 19,5 % | -8,2 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** cortes del notebook. `Sin dato` = 349 contratos sin fila de Procesos. No se aplica la regla n < 500 a tramos numéricos.

<details>
<summary>Código que genera la tabla</summary>

```python
tabla_tramos(
    Xy, "p_proveedores_invitados",
    bins=[-np.inf, 0, 4, 9, np.inf],
    labels=["0", "1-4", "5-9", "10+"],
)
```

</details>

### Tabla 6.2.2.b. Tasa de y por proveedores invitados (train 2019–2023)

**Qué muestra:** Tasa de y por tramo de `p_proveedores_invitados`.

**Subconjunto:** train 2019–2023 (n = 261.831; y = 28,8 %).

| p_proveedores_invitados | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| 0 | 255.976 | 97,8 % | 28,7 % | -0,2 pp |
| 1-4 | 3.040 | 1,2 % | 31,2 % | +2,3 pp |
| 5-9 | 722 | 0,3 % | 39,9 % | +11,1 pp |
| 10+ | 2.087 | 0,8 % | 41,1 % | +12,3 pp |
| Sin dato | 6 | 0,0 % | 16,7 % | -12,2 pp |
| Total | 261.831 | 100,0 % | 28,8 % | 0,0 pp |

**Nota:** Mismos cortes del notebook. `Sin dato` = contratos sin fila de Procesos. No se aplica la regla n < 500 a tramos numéricos.

**Dirección frente a 2019–2025:** Se mantiene: la tasa sube con los invitados, de 28,7 % con 0 a 41,1 % con 10+ (2019–2025: 27,5 % a 41,7 %).

<details>
<summary>Código que genera la tabla</summary>

```python
train = Xy[Xy["anio_firma"] <= 2023]
tabla_tramos(
    train, "p_proveedores_invitados",
    bins=[-np.inf, 0, 4, 9, np.inf],
    labels=["0", "1-4", "5-9", "10+"],
)
```

</details>

## §6.3 H3. La modalidad influye

**Subconjunto:** todos los contratos 2019–2025 (n = 452.494; y = 27,6 %). No usa test 2024.

### Tabla 6.3.1. Tasa de y por modalidad de contratación

**Qué muestra:** Tasa de y por modalidad de contratación, ordenada por `% y=1`.

| modalidad_de_contratacion | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| Licitación pública Obra Publica | 810 | 0,2 % | 79,3 % | +51,6 pp |
| Concurso de méritos abierto | 1.284 | 0,3 % | 65,9 % | +38,3 pp |
| Licitación pública | 1.184 | 0,3 % | 59,0 % | +31,4 pp |
| Otras | 243 | 0,1 % | 51,4 % | +23,8 pp |
| Selección Abreviada de Menor Cuantía | 4.623 | 1,0 % | 41,5 % | +13,9 pp |
| Contratación régimen especial | 60.145 | 13,3 % | 40,7 % | +13,1 pp |
| Selección abreviada subasta inversa | 4.141 | 0,9 % | 36,5 % | +8,9 pp |
| Contratación régimen especial (con ofertas) | 3.463 | 0,8 % | 36,1 % | +8,5 pp |
| Contratación Directa (con ofertas) | 4.833 | 1,1 % | 31,7 % | +4,1 pp |
| Contratación directa | 347.300 | 76,8 % | 25,2 % | -2,4 pp |
| Mínima cuantía | 24.468 | 5,4 % | 18,0 % | -9,6 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** se parte de las 12 modalidades más frecuentes del notebook. De ellas, 2 tienen n < 500 y se agrupan en `Otras` junto con las 3 que quedaron fuera del top 12. `Otras` agrupa 5 categorías (243 contratos). En el notebook, el resto del top 12 se rotulaba `OTROS`.

<details>
<summary>Código que genera la tabla</summary>

```python
tabla_estandar(
    Xy, "modalidad_de_contratacion",
    top=12, min_n=500, orden="tasa",
)
```

</details>

### Tabla 6.3.1.b. Tasa de y por modalidad de contratación (train 2019–2023)

**Qué muestra:** Tasa de y por modalidad de contratación, ordenada por `% y=1`.

**Subconjunto:** train 2019–2023 (n = 261.831; y = 28,8 %).

| modalidad_de_contratacion | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| Licitación pública Obra Publica | 561 | 0,2 % | 81,3 % | +52,4 pp |
| Concurso de méritos abierto | 869 | 0,3 % | 67,5 % | +38,7 pp |
| Licitación pública | 749 | 0,3 % | 59,7 % | +30,8 pp |
| Otras | 174 | 0,1 % | 57,5 % | +28,6 pp |
| Contratación régimen especial | 30.663 | 11,7 % | 47,0 % | +18,1 pp |
| Selección Abreviada de Menor Cuantía | 2.951 | 1,1 % | 40,2 % | +11,4 pp |
| Contratación régimen especial (con ofertas) | 1.908 | 0,7 % | 39,4 % | +10,6 pp |
| Selección abreviada subasta inversa | 2.523 | 1,0 % | 34,9 % | +6,1 pp |
| Contratación Directa (con ofertas) | 2.897 | 1,1 % | 31,3 % | +2,5 pp |
| Contratación directa | 203.256 | 77,6 % | 26,1 % | -2,7 pp |
| Mínima cuantía | 15.280 | 5,8 % | 17,2 % | -11,6 pp |
| Total | 261.831 | 100,0 % | 28,8 % | 0,0 pp |

**Nota:** Se parte de las 12 modalidades más frecuentes del train. Las categorías con n < 500 y las que quedan fuera del top 12 se agrupan en `Otras` (5 categorías; 174 contratos).

**Dirección frente a 2019–2025:** Se mantiene la dirección general; cambia el orden entre régimen especial (47,0 %) y selección abreviada (40,2 %).

<details>
<summary>Código que genera la tabla</summary>

```python
train = Xy[Xy["anio_firma"] <= 2023]
tabla_estandar(
    train, "modalidad_de_contratacion",
    top=12, min_n=500, orden="tasa",
)
```

</details>

## §6.4 H4. La anualidad presupuestal presiona el plazo

**Subconjunto:** todos los contratos 2019–2025 (n = 452.494; y = 27,6 %) en las tablas 6.4.1 y 6.4.2. El calendario usa solo contratos firmados entre octubre de 2021 y diciembre de 2023 (n = 166.590; y = 29,2 %).

### Tabla 6.4.1. Tasa de y por mes de firma

**Qué muestra:** Tasa de y por mes de firma (1 = enero, 12 = diciembre).

| mes | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| 1 | 82.120 | 18,1 % | 33,9 % | +6,2 pp |
| 2 | 63.897 | 14,1 % | 34,5 % | +6,9 pp |
| 3 | 35.669 | 7,9 % | 32,1 % | +4,5 pp |
| 4 | 24.962 | 5,5 % | 28,1 % | +0,5 pp |
| 5 | 26.881 | 5,9 % | 29,2 % | +1,6 pp |
| 6 | 26.565 | 5,9 % | 29,6 % | +2,0 pp |
| 7 | 33.062 | 7,3 % | 25,2 % | -2,5 pp |
| 8 | 34.231 | 7,6 % | 24,6 % | -3,0 pp |
| 9 | 35.200 | 7,8 % | 19,4 % | -8,2 pp |
| 10 | 33.551 | 7,4 % | 18,1 % | -9,5 pp |
| 11 | 33.124 | 7,3 % | 16,4 % | -11,3 pp |
| 12 | 23.232 | 5,1 % | 25,3 % | -2,3 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** los 12 meses tienen n ≥ 500; no se agrupó ninguna categoría.

<details>
<summary>Código que genera la tabla</summary>

```python
Xy_mes = Xy.assign(mes=Xy["mes_firma"].astype(int).astype(str))
tabla_estandar(
    Xy_mes, "mes", agrupar_pequenas=False, orden="natural",
    orden_natural=[str(i) for i in range(1, 13)],
)
```

</details>

### Tabla 6.4.1.b. Tasa de y por mes de firma (train 2019–2023)

**Qué muestra:** Tasa de y por mes de firma (1 = enero, 12 = diciembre).

**Subconjunto:** train 2019–2023 (n = 261.831; y = 28,8 %).

| mes | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| 1 | 57.494 | 22,0 % | 33,8 % | +4,9 pp |
| 2 | 36.205 | 13,8 % | 36,4 % | +7,5 pp |
| 3 | 18.677 | 7,1 % | 35,4 % | +6,6 pp |
| 4 | 11.369 | 4,3 % | 30,2 % | +1,4 pp |
| 5 | 13.635 | 5,2 % | 32,7 % | +3,9 pp |
| 6 | 16.701 | 6,4 % | 32,9 % | +4,0 pp |
| 7 | 19.044 | 7,3 % | 27,2 % | -1,7 pp |
| 8 | 19.979 | 7,6 % | 24,9 % | -3,9 pp |
| 9 | 18.541 | 7,1 % | 18,3 % | -10,5 pp |
| 10 | 16.814 | 6,4 % | 16,1 % | -12,7 pp |
| 11 | 19.621 | 7,5 % | 16,2 % | -12,6 pp |
| 12 | 13.751 | 5,3 % | 25,5 % | -3,4 pp |
| Total | 261.831 | 100,0 % | 28,8 % | 0,0 pp |

**Nota:** Los 12 meses tienen n ≥ 500; no se agrupó ninguna categoría.

**Dirección frente a 2019–2025:** Se mantiene: enero–junio por encima de la base y septiembre–noviembre por debajo; el máximo sigue en febrero (36,4 % vs 34,5 %).

<details>
<summary>Código que genera la tabla</summary>

```python
train = Xy[Xy["anio_firma"] <= 2023]
train_mes = train.assign(mes=train["mes_firma"].astype(int).astype(str))
tabla_estandar(
    train_mes, "mes", agrupar_pequenas=False, orden="natural",
    orden_natural=[str(i) for i in range(1, 13)],
)
```

</details>

### Tabla 6.4.2. Tasa de y por último trimestre

**Qué muestra:** Tasa de y para contratos firmados en el último trimestre (octubre–diciembre) frente al resto.

| trimestre | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| Resto | 362.587 | 80,1 % | 29,7 % | +2,1 pp |
| Último trimestre | 89.907 | 19,9 % | 19,3 % | -8,3 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** `firma_ultimo_trimestre = 1` para octubre–diciembre. Ambas categorías tienen n ≥ 500.

<details>
<summary>Código que genera la tabla</summary>

```python
Xy_trim = Xy.assign(
    trimestre=Xy["firma_ultimo_trimestre"].map(
        {0: "Resto", 1: "Último trimestre"}
    )
)
tabla_estandar(
    Xy_trim, "trimestre", agrupar_pequenas=False,
    orden="natural", orden_natural=["Resto", "Último trimestre"],
)
```

</details>

### Tabla 6.4.2.b. Tasa de y por último trimestre (train 2019–2023)

**Qué muestra:** Tasa de y para contratos firmados en el último trimestre (octubre–diciembre) frente al resto.

**Subconjunto:** train 2019–2023 (n = 261.831; y = 28,8 %).

| trimestre | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| Resto | 211.645 | 80,8 % | 31,2 % | +2,4 pp |
| Último trimestre | 50.186 | 19,2 % | 18,7 % | -10,1 pp |
| Total | 261.831 | 100,0 % | 28,8 % | 0,0 pp |

**Nota:** `firma_ultimo_trimestre = 1` para octubre–diciembre. Ambas categorías tienen n ≥ 500.

**Dirección frente a 2019–2025:** Se mantiene y se acentúa: último trimestre 18,7 % vs resto 31,2 % (2019–2025: 19,3 % vs 29,7 %).

<details>
<summary>Código que genera la tabla</summary>

```python
train = Xy[Xy["anio_firma"] <= 2023]
train_trim = train.assign(
    trimestre=train["firma_ultimo_trimestre"].map(
        {0: "Resto", 1: "Último trimestre"}
    )
)
tabla_estandar(
    train_trim, "trimestre", agrupar_pequenas=False,
    orden="natural", orden_natural=["Resto", "Último trimestre"],
)
```

</details>

### Calendario octubre 2021–diciembre 2023

**Subconjunto:** contratos con `fecha_de_firma` entre 2021-10-01 y 2023-12-31 (n = 166.590; y = 29,2 %). Esta tabla usa un periodo distinto de 2019–2025.

#### Tabla 6.4.3. Contratos y tasa de y por mes

**Qué muestra:** Contratos, % del periodo, % y=1 y diferencia en pp frente a la tasa del periodo, por mes.

| mes | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| 2021-10 | 4.097 | 2,5 % | 17,3 % | -11,9 pp |
| 2021-11 | 4.370 | 2,6 % | 19,0 % | -10,2 pp |
| 2021-12 | 3.275 | 2,0 % | 32,8 % | +3,6 pp |
| 2022-01 | 29.457 | 17,7 % | 36,4 % | +7,2 pp |
| 2022-02 | 4.137 | 2,5 % | 41,5 % | +12,3 pp |
| 2022-03 | 932 | 0,6 % | 32,5 % | +3,3 pp |
| 2022-04 | 798 | 0,5 % | 41,1 % | +11,9 pp |
| 2022-05 | 855 | 0,5 % | 35,6 % | +6,4 pp |
| 2022-06 | 1.244 | 0,7 % | 40,0 % | +10,8 pp |
| 2022-07 | 5.460 | 3,3 % | 28,4 % | -0,8 pp |
| 2022-08 | 7.250 | 4,4 % | 25,6 % | -3,6 pp |
| 2022-09 | 6.220 | 3,7 % | 18,1 % | -11,1 pp |
| 2022-10 | 5.299 | 3,2 % | 13,6 % | -15,6 pp |
| 2022-11 | 5.498 | 3,3 % | 15,7 % | -13,5 pp |
| 2022-12 | 3.941 | 2,4 % | 23,1 % | -6,1 pp |
| 2023-01 | 12.579 | 7,6 % | 34,8 % | +5,6 pp |
| 2023-02 | 14.365 | 8,6 % | 38,2 % | +9,0 pp |
| 2023-03 | 8.461 | 5,1 % | 37,1 % | +8,0 pp |
| 2023-04 | 4.387 | 2,6 % | 30,0 % | +0,8 pp |
| 2023-05 | 6.669 | 4,0 % | 33,0 % | +3,8 pp |
| 2023-06 | 10.324 | 6,2 % | 33,6 % | +4,5 pp |
| 2023-07 | 4.784 | 2,9 % | 25,0 % | -4,2 pp |
| 2023-08 | 4.204 | 2,5 % | 22,2 % | -7,0 pp |
| 2023-09 | 4.178 | 2,5 % | 17,8 % | -11,4 pp |
| 2023-10 | 4.004 | 2,4 % | 17,8 % | -11,4 pp |
| 2023-11 | 6.052 | 3,6 % | 14,0 % | -15,2 pp |
| 2023-12 | 3.750 | 2,3 % | 18,5 % | -10,7 pp |
| Total | 166.590 | 100,0 % | 29,2 % | 0,0 pp |

**Nota:** 27 meses en orden cronológico (formato aaaa-mm). Tasa base del periodo: 29,2 %. No se agrupó ningún mes.

<details>
<summary>Código que genera la tabla</summary>

```python
periodo = Xy.loc[
    (Xy["fecha_de_firma"] >= "2021-10-01") & (Xy["fecha_de_firma"] < "2024-01-01")
].copy()
periodo["mes"] = periodo["fecha_de_firma"].dt.to_period("M").astype(str)
meses = sorted(periodo["mes"].unique())
tabla_estandar(
    periodo, "mes", agrupar_pequenas=False,
    orden="natural", orden_natural=meses,
)
```

</details>

#### Tabla 6.4.4. Contratos por mes y modalidad

**Qué muestra:** Número de contratos por mes y modalidad en octubre 2021–diciembre 2023; las celdas son conteos, no tasas.

| mes | Contratación directa | Contratación régimen especial | Mínima cuantía | Selección Abreviada de Menor Cuantía | Contratación Directa (con ofertas) | Selección abreviada subasta inversa | Contratación régimen especial (con ofertas) | Otras | Total |
|---|---|---|---|---|---|---|---|---|---|
| 2021-10 | 3.237 | 327 | 276 | 64 | 81 | 56 | 23 | 33 | 4.097 |
| 2021-11 | 3.212 | 250 | 450 | 81 | 225 | 67 | 38 | 47 | 4.370 |
| 2021-12 | 2.011 | 317 | 496 | 90 | 53 | 131 | 68 | 109 | 3.275 |
| 2022-01 | 28.227 | 1.036 | 40 | 2 | 84 | 7 | 40 | 21 | 29.457 |
| 2022-02 | 2.580 | 1.266 | 188 | 15 | 18 | 11 | 19 | 40 | 4.137 |
| 2022-03 | 170 | 228 | 375 | 52 | 6 | 22 | 48 | 31 | 932 |
| 2022-04 | 124 | 185 | 312 | 56 | 7 | 36 | 42 | 36 | 798 |
| 2022-05 | 115 | 128 | 387 | 79 | 9 | 47 | 36 | 54 | 855 |
| 2022-06 | 350 | 270 | 372 | 82 | 28 | 54 | 37 | 51 | 1.244 |
| 2022-07 | 4.485 | 382 | 281 | 80 | 68 | 60 | 41 | 63 | 5.460 |
| 2022-08 | 5.869 | 782 | 282 | 69 | 82 | 64 | 53 | 49 | 7.250 |
| 2022-09 | 4.863 | 734 | 306 | 64 | 69 | 54 | 57 | 73 | 6.220 |
| 2022-10 | 3.981 | 761 | 295 | 59 | 65 | 49 | 50 | 39 | 5.299 |
| 2022-11 | 3.876 | 898 | 384 | 65 | 87 | 74 | 55 | 59 | 5.498 |
| 2022-12 | 2.067 | 778 | 520 | 100 | 124 | 115 | 106 | 131 | 3.941 |
| 2023-01 | 10.113 | 2.342 | 47 | 7 | 28 | 2 | 24 | 16 | 12.579 |
| 2023-02 | 12.843 | 1.217 | 143 | 21 | 53 | 8 | 45 | 35 | 14.365 |
| 2023-03 | 6.704 | 1.202 | 301 | 51 | 64 | 39 | 55 | 45 | 8.461 |
| 2023-04 | 3.189 | 712 | 261 | 48 | 53 | 41 | 52 | 31 | 4.387 |
| 2023-05 | 5.171 | 886 | 321 | 69 | 82 | 44 | 59 | 37 | 6.669 |
| 2023-06 | 8.390 | 1.088 | 370 | 88 | 227 | 54 | 62 | 45 | 10.324 |
| 2023-07 | 3.438 | 749 | 313 | 85 | 34 | 57 | 53 | 55 | 4.784 |
| 2023-08 | 2.908 | 643 | 366 | 90 | 18 | 54 | 57 | 68 | 4.204 |
| 2023-09 | 2.974 | 528 | 358 | 91 | 28 | 62 | 67 | 70 | 4.178 |
| 2023-10 | 2.692 | 629 | 375 | 98 | 24 | 74 | 45 | 67 | 4.004 |
| 2023-11 | 4.533 | 762 | 449 | 77 | 56 | 80 | 41 | 54 | 6.052 |
| 2023-12 | 1.928 | 797 | 593 | 101 | 83 | 106 | 44 | 98 | 3.750 |
| Total | 130.050 | 19.897 | 8.861 | 1.784 | 1.756 | 1.468 | 1.317 | 1.457 | 166.590 |

**Nota:** se muestran las 7 modalidades más frecuentes del periodo. `Otras` agrupa 8 modalidades. Total del periodo: 166.590 contratos.

<details>
<summary>Código que genera la tabla</summary>

```python
counts = periodo["modalidad_de_contratacion"].value_counts()
top7 = list(counts.head(7).index)
periodo["mod_grupo"] = periodo["modalidad_de_contratacion"].where(
    periodo["modalidad_de_contratacion"].isin(top7), "Otras"
)
ct = pd.crosstab(
    periodo["mes"], periodo["mod_grupo"],
    margins=True, margins_name="Total",
)
ct = ct.reindex(
    index=meses + ["Total"],
    columns=top7 + ["Otras", "Total"], fill_value=0,
)
```

</details>

#### Tabla 6.4.5. Contratos y tasa de y por modalidad en el periodo

**Qué muestra:** Contratos, % del periodo, % y=1 y diferencia en pp frente a la tasa del periodo, por modalidad.

| mod_grupo | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| Otras | 1.457 | 0,9 % | 69,9 % | +40,7 pp |
| Selección Abreviada de Menor Cuantía | 1.784 | 1,1 % | 40,8 % | +11,6 pp |
| Contratación régimen especial (con ofertas) | 1.317 | 0,8 % | 38,0 % | +8,8 pp |
| Contratación régimen especial | 19.897 | 11,9 % | 37,9 % | +8,7 pp |
| Selección abreviada subasta inversa | 1.468 | 0,9 % | 35,8 % | +6,6 pp |
| Contratación Directa (con ofertas) | 1.756 | 1,1 % | 32,9 % | +3,7 pp |
| Contratación directa | 130.050 | 78,1 % | 27,8 % | -1,4 pp |
| Mínima cuantía | 8.861 | 5,3 % | 18,7 % | -10,5 pp |
| Total | 166.590 | 100,0 % | 29,2 % | 0,0 pp |

**Nota:** `Otras` agrupa las modalidades fuera del top 7 del periodo (8 categorías). Tasa base del periodo: 29,2 %.

<details>
<summary>Código que genera la tabla</summary>

```python
tabla_estandar(
    periodo, "mod_grupo", min_n=500,
    agrupar_pequenas=True, orden="tasa",
)
```

</details>

## §6.5 H5. Los plazos subestimados terminan ampliándose

**Subconjunto:** todos los contratos 2019–2025 (n = 452.494; y = 27,6 %). No usa test 2024.

### Tabla 6.5.1. Tasa de y por duración relativa a la mediana de la familia

**Qué muestra:** Tasa de y por `p_dur_dias / mediana de la familia UNSPSC`.

| duracion_relativa | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| <=0 | 24.099 | 5,3 % | 24,5 % | -3,1 pp |
| (0, 0.5] | 94.724 | 20,9 % | 19,9 % | -7,7 pp |
| (0.5, 1] | 132.540 | 29,3 % | 27,2 % | -0,4 pp |
| (1, 1.5] | 84.427 | 18,7 % | 30,2 % | +2,5 pp |
| (1.5, 2] | 55.873 | 12,3 % | 40,5 % | +12,8 pp |
| (2, 3] | 47.255 | 10,4 % | 26,3 % | -1,4 pp |
| >3 | 8.951 | 2,0 % | 24,6 % | -3,1 pp |
| Sin dato | 4.625 | 1,0 % | 31,5 % | +3,9 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** cortes del notebook. La mediana de referencia se calcula con 2019–2023 y duración > 0. `Sin dato` = 4.625 contratos sin duración o sin mediana de familia. No se aplica la regla n < 500 a tramos numéricos.

<details>
<summary>Código que genera la tabla</summary>

```python
dur_h5 = eda.duracion_relativa_objeto(Xy, anio_referencia=2023)
Xy_h5 = Xy.assign(duracion_relativa=dur_h5["duracion_relativa"])
tabla_tramos(
    Xy_h5, "duracion_relativa",
    bins=[-np.inf, 0, 0.5, 1, 1.5, 2, 3, np.inf],
    labels=["<=0", "(0, 0.5]", "(0.5, 1]", "(1, 1.5]",
            "(1.5, 2]", "(2, 3]", ">3"],
)
```

</details>

### Tabla 6.5.1.b. Tasa de y por duración relativa a la mediana de la familia (train 2019–2023)

**Qué muestra:** Tasa de y por `p_dur_dias / mediana de la familia UNSPSC`.

**Subconjunto:** train 2019–2023 (n = 261.831; y = 28,8 %).

| duracion_relativa | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| <=0 | 13.517 | 5,2 % | 25,2 % | -3,6 pp |
| (0, 0.5] | 51.465 | 19,7 % | 20,5 % | -8,4 pp |
| (0.5, 1] | 74.858 | 28,6 % | 28,4 % | -0,4 pp |
| (1, 1.5] | 50.688 | 19,4 % | 31,6 % | +2,8 pp |
| (1.5, 2] | 33.771 | 12,9 % | 41,1 % | +12,2 pp |
| (2, 3] | 29.503 | 11,3 % | 27,8 % | -1,1 pp |
| >3 | 5.836 | 2,2 % | 27,7 % | -1,2 pp |
| Sin dato | 2.193 | 0,8 % | 26,0 % | -2,8 pp |
| Total | 261.831 | 100,0 % | 28,8 % | 0,0 pp |

**Nota:** Mismos cortes del notebook. La mediana de referencia se calcula con el train (2019–2023) y duración > 0. `Sin dato` = contratos sin duración o sin mediana de familia. No se aplica la regla n < 500 a tramos numéricos.

**Dirección frente a 2019–2025:** Se mantiene: mínimo en (0, 0,5] (20,5 % vs 19,9 %) y máximo en (1,5, 2] (41,1 % vs 40,5 %).

<details>
<summary>Código que genera la tabla</summary>

```python
train = Xy[Xy["anio_firma"] <= 2023]
dur_h5 = eda.duracion_relativa_objeto(train, anio_referencia=2023)
train_h5 = train.assign(duracion_relativa=dur_h5["duracion_relativa"])
tabla_tramos(
    train_h5, "duracion_relativa",
    bins=[-np.inf, 0, 0.5, 1, 1.5, 2, 3, np.inf],
    labels=["<=0", "(0, 0.5]", "(0.5, 1]", "(1, 1.5]",
            "(1.5, 2]", "(2, 3]", ">3"],
)
```

</details>

### Tabla 6.5.2. Tasa de y por duración planeada absoluta

**Qué muestra:** Tasa de y por tramo de `p_dur_dias` (duración planeada en días).

| p_dur_dias | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| <=0 | 24.566 | 5,4 % | 24,7 % | -2,9 pp |
| 1-30 | 42.517 | 9,4 % | 20,5 % | -7,1 pp |
| 31-90 | 108.000 | 23,9 % | 23,1 % | -4,6 pp |
| 91-180 | 144.602 | 32,0 % | 27,9 % | +0,2 pp |
| 181-365 | 126.781 | 28,0 % | 33,7 % | +6,1 pp |
| 366-730 | 2.951 | 0,7 % | 57,1 % | +29,5 pp |
| >730 | 2.278 | 0,5 % | 22,1 % | -5,5 pp |
| Sin dato | 799 | 0,2 % | 15,0 % | -12,6 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** cortes del notebook. `<=0` incluye duración 0 y negativa; `Sin dato` = 799 contratos sin duración. No se aplica la regla n < 500 a tramos numéricos.

<details>
<summary>Código que genera la tabla</summary>

```python
tabla_tramos(
    Xy, "p_dur_dias",
    bins=[-np.inf, 0, 30, 90, 180, 365, 730, np.inf],
    labels=["<=0", "1-30", "31-90", "91-180", "181-365",
            "366-730", ">730"],
)
```

</details>

### Tabla 6.5.2.b. Tasa de y por duración planeada absoluta (train 2019–2023)

**Qué muestra:** Tasa de y por tramo de `p_dur_dias` (duración planeada en días).

**Subconjunto:** train 2019–2023 (n = 261.831; y = 28,8 %).

| p_dur_dias | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| <=0 | 13.712 | 5,2 % | 25,3 % | -3,5 pp |
| 1-30 | 23.487 | 9,0 % | 22,8 % | -6,0 pp |
| 31-90 | 58.498 | 22,3 % | 23,1 % | -5,8 pp |
| 91-180 | 83.890 | 32,0 % | 28,9 % | +0,1 pp |
| 181-365 | 78.359 | 29,9 % | 34,7 % | +5,8 pp |
| 366-730 | 2.222 | 0,8 % | 59,6 % | +30,8 pp |
| >730 | 1.381 | 0,5 % | 28,5 % | -0,3 pp |
| Sin dato | 282 | 0,1 % | 11,3 % | -17,5 pp |
| Total | 261.831 | 100,0 % | 28,8 % | 0,0 pp |

**Nota:** Mismos cortes del notebook. `<=0` incluye duración 0 y negativa; `Sin dato` = contratos sin duración. No se aplica la regla n < 500 a tramos numéricos.

**Dirección frente a 2019–2025:** Se mantiene: sube hasta 366–730 (59,6 % vs 57,1 %) y cae en >730 (28,5 % vs 22,1 %).

<details>
<summary>Código que genera la tabla</summary>

```python
train = Xy[Xy["anio_firma"] <= 2023]
tabla_tramos(
    train, "p_dur_dias",
    bins=[-np.inf, 0, 30, 90, 180, 365, 730, np.inf],
    labels=["<=0", "1-30", "31-90", "91-180", "181-365",
            "366-730", ">730"],
)
```

</details>

## §6.6 H6. Las entidades tienen hábitos

**Subconjunto:** test 2024 (n = 88.546; y = 24,65 %). La tasa histórica de cada entidad se calcula solo con 2019–2023. Esta sección usa test 2024, no todos los años.

### Tabla 6.6.1. Tasa de y en 2024 por grupo de tasa histórica de la entidad

**Qué muestra:** Tasa de y en 2024 por grupo de tasa histórica 2019–2023 de la entidad, con n, % del total, diferencia en pp, número de entidades y rango de la tasa histórica.

| grupo | n | % del total | % y=1 | diferencia (pp) | n entidades | tasa hist. mín | mediana | máx |
|---|---|---|---|---|---|---|---|---|
| Sin historia | 1.833 | 2,1 % | 11,5 % | -13,1 pp | 317 | — | — | — |
| Tasa 0 | 6.460 | 7,3 % | 11,4 % | -13,3 pp | 553 | 0,0 % | 0,0 % | 0,0 % |
| T1 | 21.231 | 24,0 % | 14,3 % | -10,4 pp | 416 | 0,4 % | 10,3 % | 16,4 % |
| T2 | 31.741 | 35,8 % | 20,5 % | -4,1 pp | 423 | 16,5 % | 22,6 % | 30,6 % |
| T3 | 27.281 | 30,8 % | 41,6 % | +16,9 pp | 377 | 30,7 % | 42,9 % | 100,0 % |
| Total | 88.546 | 100,0 % | 24,7 % | 0,0 pp | 2.086 | — | — | — |

**Nota:** `Sin historia` = entidades sin contratos en 2019–2023; `Tasa 0` = tasa histórica 0; `T1`–`T3` = terciles de la tasa histórica > 0, armados sobre el rank para que los empates no colapsen. `T1` es el tercil de menor tasa y `T3` el de mayor. La columna `n entidades` cuenta las entidades observadas en 2024; el resumen 6.6.2 cuenta todas las entidades con historia.

<details>
<summary>Código que genera la tabla</summary>

```python
tabla_h6, hist_h6 = eda.tasa_entidad_historica(base, anio_objetivo=2024)
n_total = int(tabla_h6["n"].sum())
base_rate = 100 * int(tabla_h6["positivos"].sum()) / n_total
for r in tabla_h6.to_dict("records"):
    tasa = 100 * r["tasa"]
    print(r["grupo_hist"], r["n"], 100 * r["n"] / n_total, tasa,
          tasa - base_rate, r["n_entidades"], r["tasa_hist_min"],
          r["tasa_hist_mediana"], r["tasa_hist_max"])
```

</details>

### Tabla 6.6.2. Resumen por entidad de la tasa histórica

**Qué muestra:** Número de entidades y rango de la tasa histórica 2019–2023 dentro de cada grupo.

| grupo | entidades | tasa mín | mediana | máx |
|---|---|---|---|---|
| T1 | 439 | 0,4 % | 10,3 % | 16,4 % |
| T2 | 439 | 16,5 % | 22,6 % | 30,6 % |
| T3 | 439 | 30,7 % | 42,9 % | 100,0 % |
| Tasa 0 | 1.017 | 0,0 % | 0,0 % | 0,0 % |

**Nota:** cuenta todas las entidades con historia 2019–2023, no solo las que aparecen en 2024. `Sin historia` no entra porque no tiene tasa histórica.

<details>
<summary>Código que genera la tabla</summary>

```python
resumen = (
    hist_h6.groupby("grupo", observed=False)
    .agg(entidades=("tasa", "size"), tasa_min=("tasa", "min"),
         tasa_mediana=("tasa", "median"), tasa_max=("tasa", "max"))
    .reset_index()
)
```

</details>

## §6.7 H7. El tipo de objeto importa

**Subconjunto:** todos los contratos 2019–2025 (n = 452.494; y = 27,6 %). No usa test 2024.

### Tabla 6.7.1. Tasa de y por segmento UNSPSC

**Qué muestra:** Tasa de y por segmento UNSPSC, ordenada por `% y=1`.

| unspsc_segmento | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| 72 | 7.355 | 1,6 % | 38,9 % | +11,2 pp |
| 85 | 26.352 | 5,8 % | 38,5 % | +10,9 pp |
| Sin dato | 3.806 | 0,8 % | 35,0 % | +7,3 pp |
| 93 | 21.784 | 4,8 % | 31,9 % | +4,3 pp |
| 81 | 11.535 | 2,5 % | 28,2 % | +0,5 pp |
| 80 | 323.045 | 71,4 % | 27,0 % | -0,7 pp |
| 78 | 3.344 | 0,7 % | 25,4 % | -2,3 pp |
| 84 | 3.811 | 0,8 % | 25,3 % | -2,3 pp |
| 86 | 9.692 | 2,1 % | 23,2 % | -4,5 pp |
| 77 | 3.605 | 0,8 % | 22,8 % | -4,9 pp |
| Otras | 38.165 | 8,4 % | 22,2 % | -5,4 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** se conservan los 10 segmentos más frecuentes del notebook. `Otras` agrupa 47 categorías (38.165 contratos). `Sin dato` = 3.806 contratos sin UNSPSC.

<details>
<summary>Código que genera la tabla</summary>

```python
tabla_estandar(
    Xy, "unspsc_segmento", top=10, min_n=500, orden="tasa",
)
```

</details>

### Tabla 6.7.1.b. Tasa de y por segmento UNSPSC (train 2019–2023)

**Qué muestra:** Tasa de y por segmento UNSPSC, ordenada por `% y=1`.

**Subconjunto:** train 2019–2023 (n = 261.831; y = 28,8 %).

| unspsc_segmento | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| 85 | 13.544 | 5,2 % | 40,3 % | +11,5 pp |
| 72 | 4.408 | 1,7 % | 38,6 % | +9,8 pp |
| 93 | 12.528 | 4,8 % | 34,3 % | +5,5 pp |
| 81 | 7.347 | 2,8 % | 29,8 % | +1,0 pp |
| 80 | 186.680 | 71,3 % | 28,5 % | -0,4 pp |
| 84 | 2.184 | 0,8 % | 27,5 % | -1,3 pp |
| 78 | 2.017 | 0,8 % | 26,4 % | -2,4 pp |
| 86 | 6.077 | 2,3 % | 25,7 % | -3,2 pp |
| Otras | 22.768 | 8,7 % | 22,7 % | -6,2 pp |
| 77 | 2.323 | 0,9 % | 21,2 % | -7,6 pp |
| 82 | 1.955 | 0,7 % | 17,7 % | -11,1 pp |
| Total | 261.831 | 100,0 % | 28,8 % | 0,0 pp |

**Nota:** Se conservan los 10 segmentos más frecuentes del train. `Otras` agrupa 47 categorías (22.768 contratos). `Sin dato` = contratos sin UNSPSC.

**Dirección frente a 2019–2025:** Se mantiene la dirección general; cambia el orden entre 85 y 72 (40,3 % vs 38,6 %; en 2019–2025, 38,5 % vs 38,9 %).

<details>
<summary>Código que genera la tabla</summary>

```python
train = Xy[Xy["anio_firma"] <= 2023]
tabla_estandar(
    train, "unspsc_segmento", top=10, min_n=500, orden="tasa",
)
```

</details>

### Tabla 6.7.2. Tasa de y por familia UNSPSC

**Qué muestra:** Tasa de y por familia UNSPSC, ordenada por `% y=1`.

| unspsc_familia | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| 8510 | 19.289 | 4,3 % | 43,7 % | +16,1 pp |
| 8110 | 4.715 | 1,0 % | 38,9 % | +11,2 pp |
| 9314 | 11.844 | 2,6 % | 32,9 % | +5,3 pp |
| 9315 | 8.337 | 1,8 % | 32,1 % | +4,5 pp |
| 8011 | 285.020 | 63,0 % | 27,2 % | -0,4 pp |
| 8010 | 12.386 | 2,7 % | 26,5 % | -1,2 pp |
| 8512 | 5.081 | 1,1 % | 25,0 % | -2,6 pp |
| Otras | 79.899 | 17,7 % | 24,9 % | -2,8 pp |
| 8016 | 14.181 | 3,1 % | 24,5 % | -3,2 pp |
| 8610 | 6.485 | 1,4 % | 24,1 % | -3,5 pp |
| 8012 | 5.257 | 1,2 % | 22,4 % | -5,2 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** se conservan las 10 familias más frecuentes del notebook. `Otras` agrupa 367 categorías (79.899 contratos): las que quedaron fuera del top 10 y las que tienen n < 500. Incluye `Sin dato` (3.806 contratos).

<details>
<summary>Código que genera la tabla</summary>

```python
tabla_estandar(
    Xy, "unspsc_familia", top=10, min_n=500, orden="tasa",
)
```

</details>

### Tabla 6.7.2.b. Tasa de y por familia UNSPSC (train 2019–2023)

**Qué muestra:** Tasa de y por familia UNSPSC, ordenada por `% y=1`.

**Subconjunto:** train 2019–2023 (n = 261.831; y = 28,8 %).

| unspsc_familia | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| 8510 | 9.672 | 3,7 % | 45,8 % | +17,0 pp |
| 8110 | 2.976 | 1,1 % | 41,0 % | +12,2 pp |
| 9315 | 4.715 | 1,8 % | 38,5 % | +9,7 pp |
| 9314 | 6.946 | 2,7 % | 32,7 % | +3,8 pp |
| 8011 | 162.023 | 61,9 % | 28,8 % | +0,0 pp |
| 8610 | 4.119 | 1,6 % | 28,0 % | -0,9 pp |
| 8016 | 9.241 | 3,5 % | 26,2 % | -2,7 pp |
| 8010 | 8.092 | 3,1 % | 26,1 % | -2,7 pp |
| Otras | 47.595 | 18,2 % | 25,0 % | -3,8 pp |
| 8012 | 3.471 | 1,3 % | 23,4 % | -5,5 pp |
| 8111 | 2.981 | 1,1 % | 21,6 % | -7,2 pp |
| Total | 261.831 | 100,0 % | 28,8 % | 0,0 pp |

**Nota:** Se conservan las 10 familias más frecuentes del train. `Otras` agrupa 358 categorías (47.595 contratos): las que quedaron fuera del top 10 y las que tienen n < 500. Incluye `Sin dato`.

**Dirección frente a 2019–2025:** Se mantiene la dirección general; cambian de orden 9315 y 9314 y entra 8111 en lugar de 8512.

<details>
<summary>Código que genera la tabla</summary>

```python
train = Xy[Xy["anio_firma"] <= 2023]
tabla_estandar(
    train, "unspsc_familia", top=10, min_n=500, orden="tasa",
)
```

</details>

### Tabla 6.7.3. Tasa de y por tipo de contrato

**Qué muestra:** Tasa de y por tipo de contrato, ordenada por `% y=1`.

| tipo_de_contrato | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| Interventoría | 1.043 | 0,2 % | 71,8 % | +44,2 pp |
| Obra | 4.081 | 0,9 % | 51,9 % | +24,2 pp |
| Consultoría | 889 | 0,2 % | 42,4 % | +14,8 pp |
| Otro | 16.849 | 3,7 % | 37,9 % | +10,3 pp |
| Arrendamiento de inmuebles | 3.494 | 0,8 % | 33,7 % | +6,1 pp |
| Otras | 1.400 | 0,3 % | 32,1 % | +4,4 pp |
| Prestación de servicios | 388.023 | 85,8 % | 27,1 % | -0,5 pp |
| Decreto 092 de 2017 | 15.978 | 3,5 % | 25,3 % | -2,3 pp |
| Compraventa | 8.917 | 2,0 % | 22,0 % | -5,7 pp |
| Suministros | 10.780 | 2,4 % | 21,3 % | -6,4 pp |
| Comodato | 1.040 | 0,2 % | 16,4 % | -11,2 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** se conservan los 10 tipos más frecuentes del notebook. `Otras` agrupa 14 categorías (1.400 contratos).

<details>
<summary>Código que genera la tabla</summary>

```python
tabla_estandar(
    Xy, "tipo_de_contrato", top=10, min_n=500, orden="tasa",
)
```

</details>

### Tabla 6.7.3.b. Tasa de y por tipo de contrato (train 2019–2023)

**Qué muestra:** Tasa de y por tipo de contrato, ordenada por `% y=1`.

**Subconjunto:** train 2019–2023 (n = 261.831; y = 28,8 %).

| tipo_de_contrato | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| Interventoría | 723 | 0,3 % | 71,9 % | +43,1 pp |
| Obra | 2.409 | 0,9 % | 53,2 % | +24,4 pp |
| Consultoría | 543 | 0,2 % | 45,3 % | +16,5 pp |
| Otro | 9.475 | 3,6 % | 36,5 % | +7,6 pp |
| Otras | 782 | 0,3 % | 36,3 % | +7,5 pp |
| Decreto 092 de 2017 | 8.313 | 3,2 % | 35,5 % | +6,6 pp |
| Arrendamiento de inmuebles | 2.235 | 0,9 % | 34,7 % | +5,9 pp |
| Prestación de servicios | 224.529 | 85,8 % | 28,2 % | -0,6 pp |
| Compraventa | 5.813 | 2,2 % | 20,5 % | -8,3 pp |
| Suministros | 6.327 | 2,4 % | 20,3 % | -8,5 pp |
| Comodato | 682 | 0,3 % | 18,6 % | -10,2 pp |
| Total | 261.831 | 100,0 % | 28,8 % | 0,0 pp |

**Nota:** Se conservan los 10 tipos más frecuentes del train. `Otras` agrupa 12 categorías (782 contratos).

**Dirección frente a 2019–2025:** Se mantiene la dirección general; Decreto 092 pasa de -2,3 pp a +6,6 pp y sube en el orden.

<details>
<summary>Código que genera la tabla</summary>

```python
train = Xy[Xy["anio_firma"] <= 2023]
tabla_estandar(
    train, "tipo_de_contrato", top=10, min_n=500, orden="tasa",
)
```

</details>

### Tabla 6.7.4. Tasa de y por longitud del objeto

**Qué muestra:** Tasa de y por tramo de `len_objeto` (longitud de la descripción del objeto, en caracteres).

| len_objeto | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| <=50 | 5.863 | 1,3 % | 58,0 % | +30,3 pp |
| 51-100 | 17.915 | 4,0 % | 34,4 % | +6,7 pp |
| 101-200 | 142.903 | 31,6 % | 25,3 % | -2,4 pp |
| 201-400 | 239.544 | 52,9 % | 28,2 % | +0,6 pp |
| >400 | 46.269 | 10,2 % | 25,4 % | -2,2 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** cortes del notebook. `len_objeto` está truncado en 500 caracteres. No se aplica la regla n < 500 a tramos numéricos.

<details>
<summary>Código que genera la tabla</summary>

```python
tabla_tramos(
    Xy, "len_objeto",
    bins=[-np.inf, 50, 100, 200, 400, np.inf],
    labels=["<=50", "51-100", "101-200", "201-400", ">400"],
)
```

</details>

### Tabla 6.7.4.b. Tasa de y por longitud del objeto (train 2019–2023)

**Qué muestra:** Tasa de y por tramo de `len_objeto` (longitud de la descripción del objeto, en caracteres).

**Subconjunto:** train 2019–2023 (n = 261.831; y = 28,8 %).

| len_objeto | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| <=50 | 4.385 | 1,7 % | 58,5 % | +29,7 pp |
| 51-100 | 11.832 | 4,5 % | 35,5 % | +6,7 pp |
| 101-200 | 81.194 | 31,0 % | 25,7 % | -3,2 pp |
| 201-400 | 135.815 | 51,9 % | 29,6 % | +0,8 pp |
| >400 | 28.605 | 10,9 % | 26,8 % | -2,1 pp |
| Total | 261.831 | 100,0 % | 28,8 % | 0,0 pp |

**Nota:** Mismos cortes del notebook. `len_objeto` está truncado en 500 caracteres. No se aplica la regla n < 500 a tramos numéricos.

**Dirección frente a 2019–2025:** Se mantiene: <=50 sigue siendo el máximo (58,5 % vs 58,0 %) y la relación sigue sin ser monótona.

<details>
<summary>Código que genera la tabla</summary>

```python
train = Xy[Xy["anio_firma"] <= 2023]
tabla_tramos(
    train, "len_objeto",
    bins=[-np.inf, 50, 100, 200, 400, np.inf],
    labels=["<=50", "51-100", "101-200", "201-400", ">400"],
)
```

</details>

## §7.2 Tasa de y por tipo de documento del proveedor

**Subconjunto:** todos los contratos 2019–2025 (n = 452.494; y = 27,6 %). No usa test 2024.

### Tabla 7.2.1. Tasa de y por tipo de documento

**Qué muestra:** Tasa de y por `tipodocproveedor`, ordenada por `% y=1`.

| tipodocproveedor | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| No Definido | 730 | 0,2 % | 60,5 % | +32,9 pp |
| Otro | 878 | 0,2 % | 30,8 % | +3,1 pp |
| NIT | 68.844 | 15,2 % | 28,9 % | +1,2 pp |
| Cédula de Extranjería | 607 | 0,1 % | 27,7 % | +0,0 pp |
| Cédula de Ciudadanía | 381.242 | 84,3 % | 27,3 % | -0,3 pp |
| Otras | 193 | 0,0 % | 23,3 % | -4,3 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** `Otras` agrupa 5 categorías con n < 500 (193 contratos). Solo se usa el tipo de documento; nunca la identidad del proveedor.

<details>
<summary>Código que genera la tabla</summary>

```python
tabla_estandar(
    base, "tipodocproveedor", min_n=500, orden="tasa",
)
```

</details>

### Tabla 7.2.1.b. Tasa de y por tipo de documento (train 2019–2023)

**Qué muestra:** Tasa de y por `tipodocproveedor`, ordenada por `% y=1`.

**Subconjunto:** train 2019–2023 (n = 261.831; y = 28,8 %).

| tipodocproveedor | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| Otras | 950 | 0,4 % | 43,7 % | +14,8 pp |
| Otro | 564 | 0,2 % | 29,6 % | +0,8 pp |
| Cédula de Ciudadanía | 219.724 | 83,9 % | 28,8 % | +0,0 pp |
| NIT | 40.593 | 15,5 % | 28,7 % | -0,1 pp |
| Total | 261.831 | 100,0 % | 28,8 % | 0,0 pp |

**Nota:** `Otras` agrupa 7 categorías con n < 500 (950 contratos). Solo se usa el tipo de documento; nunca la identidad del proveedor.

**Dirección frente a 2019–2025:** Cambia la comparación: NIT y cédula quedan casi iguales (28,7 % vs 28,8 %; en 2019–2025, 28,9 % vs 27,3 %), y `Otras` mezcla No Definido y Cédula de Extranjería, por lo que su 43,7 % no es comparable.

<details>
<summary>Código que genera la tabla</summary>

```python
train = base[base["anio_firma"] <= 2023]
tabla_estandar(
    train, "tipodocproveedor", min_n=500, orden="tasa",
)
```

</details>

### Tabla 7.2.2. Tasa de y por tipo de documento y año

**Qué muestra:** Tasa de y por tipo de documento del proveedor en cada año; cada celda muestra `% y=1` y el n entre paréntesis.

| tipo de documento | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| No Definido | 57,7 % (n=78) | 60,7 % (n=89) | 67,2 % (n=116) | 63,6 % (n=99) | 54,6 % (n=97) | 64,9 % (n=94) | 56,1 % (n=157) |
| Otro | 37,5 % (n=24) | 35,9 % (n=78) | 26,7 % (n=135) | 28,2 % (n=142) | 29,2 % (n=185) | 29,7 % (n=148) | 35,5 % (n=166) |
| NIT | 18,8 % (n=3.402) | 32,2 % (n=6.228) | 30,5 % (n=9.054) | 29,8 % (n=10.064) | 27,5 % (n=11.845) | 28,6 % (n=13.265) | 29,4 % (n=14.986) |
| Cédula de Extranjería | 62,5 % (n=16) | 24,0 % (n=50) | 22,0 % (n=82) | 28,6 % (n=105) | 29,9 % (n=117) | 22,6 % (n=124) | 31,0 % (n=113) |
| Cédula de Ciudadanía | 22,8 % (n=10.885) | 28,6 % (n=29.651) | 27,2 % (n=47.061) | 29,3 % (n=60.648) | 30,4 % (n=71.479) | 23,9 % (n=74.875) | 26,7 % (n=86.643) |
| Otras | 0,0 % (n=2) | 18,2 % (n=11) | 14,3 % (n=21) | 12,1 % (n=33) | 23,5 % (n=34) | 35,0 % (n=40) | 26,9 % (n=52) |

**Nota:** `Otras` agrupa 5 categorías con n < 500 (193 contratos en total). Las tasas anuales de `Otras` y de las categorías con n pequeño son inestables; no se muestran por separado.

<details>
<summary>Código que genera la tabla</summary>

```python
s0 = base["tipodocproveedor"].astype("object")
s0 = s0.where(s0.notna(), "Sin dato")
counts = s0.value_counts()
small = set(counts[counts < 500].index)
base_doc = base.assign(
    grupo_doc=s0.where(~s0.isin(small), "Otras")
)
evo = (
    base_doc.groupby(["grupo_doc", "anio_firma"])["y"]
    .agg(n="size", tasa="mean")
    .reset_index()
)
```

</details>

## §8 Fuga descriptiva

**Subconjunto:** todos los contratos 2019–2025 (n = 452.494) para las columnas de Contratos. Cada AUC usa las filas no nulas de su variable; el n se reporta en la tabla. No usa test 2024 y no entrena ningún modelo.

**Construcción de la tabla de trabajo.** `d` une la base v1_A con las columnas de Contratos y los agregados de Modificaciones que se describen abajo.

<details>
<summary>Construcción de d (base + Contratos + Modificaciones)</summary>

```python
base2 = base[[
    "id_contrato", "y", "y_dias", "anio_firma",
    "p_precio_base", "fecha_de_firma",
]].copy()
base2["p_dur_dias"] = X["p_dur_dias"]
raw2 = raw.drop(columns=[
    "fecha_de_firma", "fecha_de_inicio_del_contrato",
    "fecha_de_fin_del_contrato",
])
d = base2.merge(raw2, on="id_contrato", how="left", validate="one_to_one")
d["razon_dur"] = np.where(
    d["p_dur_dias"].between(1, 3650) & d["dur_contrato_dias"].between(1, 3650),
    d["dur_contrato_dias"] / d["p_dur_dias"], np.nan,
)
d["razon_valor"] = np.where(
    d["p_precio_base"].gt(0),
    d["valor_del_contrato"] / d["p_precio_base"], np.nan,
)
firma = base.set_index("id_contrato")["fecha_de_firma"]
m = mod.copy()
m["aprobacion_dias"] = (
    m["fecha_de_aprobacion"] - m["id_contrato"].map(firma)
).dt.days
m["fin_dias_mod"] = (
    m["fecha_fin_contrato"] - m["id_contrato"].map(firma)
).dt.days
agg = (
    m.groupby("id_contrato")
    .agg(
        mod_n=("identificador_modificacion", "size"),
        mod_dias_ext_sum=("dias_extendidos", "sum"),
        mod_aprobacion_max_dias=("aprobacion_dias", "max"),
        mod_fin_max_dias=("fin_dias_mod", "max"),
    )
)
d = d.merge(agg, on="id_contrato", how="left")
for c in ["mod_n", "mod_dias_ext_sum", "mod_aprobacion_max_dias"]:
    d[c] = d[c].fillna(0)
d["mod_fin_max_dias"] = d[["mod_fin_max_dias", "fin_dias"]].max(axis=1)
```

</details>

### Tabla 8.1. Tasa de y por estado actual del contrato

**Qué muestra:** Tasa de y por `estado_contrato`, ordenada por `% y=1`.

| estado_contrato | n | % del total | % y=1 | diferencia (pp) |
|---|---|---|---|---|
| Modificado | 81.391 | 18,0 % | 76,9 % | +49,2 pp |
| Otras | 390 | 0,1 % | 59,5 % | +31,9 pp |
| terminado | 73.006 | 16,1 % | 26,0 % | -1,6 pp |
| Cerrado | 166.839 | 36,9 % | 25,5 % | -2,1 pp |
| cedido | 1.802 | 0,4 % | 14,8 % | -12,8 pp |
| En ejecución | 118.069 | 26,1 % | 0,3 % | -27,3 pp |
| Aprobado | 10.997 | 2,4 % | 0,2 % | -27,4 pp |
| Total | 452.494 | 100,0 % | 27,6 % | 0,0 pp |

**Nota:** `Otras` agrupa 3 categorías con n < 500 (390 contratos): En aprobación, Suspendido, enviado Proveedor. `estado_contrato` es una columna de fuga confirmada; se usa solo para EDA.

<details>
<summary>Código que genera la tabla</summary>

```python
tabla_estandar(
    d, "estado_contrato", min_n=500, orden="tasa",
)
```

</details>

### Tabla 8.2. ROC-AUC univariado por candidato de fuga

**Qué muestra:** ROC-AUC univariado de cada candidato de fuga frente a `y` y `y_dias`. En categóricas, el score es la tasa de y por categoría calculada en la misma muestra; en numéricas, el valor observado.

| variable | tipo | n | AUC vs y | AUC vs y_dias | referencia legacy | nota |
|---|---|---|---|---|---|---|
| estado_contrato | categórica | 452.494 | 0,820 | 0,943 | 0,94 | Tasa de y por categoría en la misma muestra; estado actual. |
| fecha_de_fin_del_contrato | fecha (días desde firma) | 452.489 | 0,744 | 0,707 | 0,71 | Fecha actual del fin; incluye ampliaciones. 5 sin fecha. |
| dur_contrato_dias | numérica | 448.436 | 0,740 | 0,704 | 0,71 | Fin − inicio. 4.058 sin fecha de inicio. |
| razon_dur | cociente | 421.120 | 0,952 | 0,885 | 0,71 | Duración real / planeada, con ambas entre 1 y 3.650 días; 0,952 calculado frente a 0,71 de referencia. |
| valor_del_contrato | numérica | 452.494 | 0,675 | 0,652 | — | Valor actual; incluye adiciones. |
| razon_valor | cociente | 448.638 | 0,810 | 0,748 | — | Valor actual / precio base, con precio base > 0. |
| recursos_suma | numérica | 452.494 | 0,657 | 0,639 | — | Suma de las 6 fuentes de recursos; AUC individuales 0,498–0,537. |
| el_contrato_puede_ser_prorrogado | categórica | 452.494 | 0,563 | 0,551 | 0,561 | Tasa de y por categoría en la misma muestra. |
| dias_adicionados | numérica | 452.494 | 0,741 | 1,0000 | — | Componente de `y_dias`; AUC 1,000 frente a `y_dias` por construcción. |
| mod_n | conteo | 452.494 | 0,9996 | 0,931 | — | N.º de modificaciones; cuasi-etiqueta porque Modificaciones ya viene filtrada a extensiones publicadas. |
| mod_dias_ext_sum | numérica | 452.494 | 0,9996 | 0,930 | — | Suma de días extendidos; cuasi-etiqueta. |
| mod_aprobacion_max_dias | fecha (días desde firma) | 452.494 | 0,9996 | 0,925 | — | Máx. fecha de aprobación − firma; 0 si no hay modificación; cuasi-etiqueta. |
| mod_fin_max_dias | fecha (días desde firma) | 452.489 | 0,747 | 0,708 | 0,71 | Máx.(fin en Modificaciones, fin actual) − firma. |
| fecha_de_notificaci_n_de_prorrogaci_n | fecha | — | NO EJECUTADO | NO EJECUTADO | — | No está en `data/raw/f2/contratos`. |
| ultima_actualizacion | fecha | — | NO EJECUTADO | NO EJECUTADO | 0,74 | No está en `data/raw/f2/contratos`. |
| dias_firma_a_ult_actualizacion | numérica | — | NO EJECUTADO | NO EJECUTADO | 0,74 | No está en `data/raw/f2/contratos`. |
| duraci_n_del_contrato | numérica | — | NO EJECUTADO | NO EJECUTADO | — | En raw solo existe la unidad (`Dia(s)`, `No definido`); no hay valor numérico. |
| valor_modificacion | numérica | — | NO EJECUTADO | NO EJECUTADO | — | No descargada de Modificaciones. |
| pagos, facturado, amortizado, pendientes y saldos (8 columnas) | grupo | — | NO EJECUTADO | NO EJECUTADO | 0,63 | No descargadas. |
| estado_del_procedimiento, visualizaciones_del, fase y otras (Procesos) | grupo | — | NO EJECUTADO | NO EJECUTADO | — | Procesos no descargado en f2. |
| liquidaci_n, reversion y sus fechas (6 columnas) | grupo | — | NO EJECUTADO | NO EJECUTADO | — | No descargadas. |

**Nota:** AUC descriptivo, no una métrica de validación; no hay train/test ni modelo. Las filas `NO EJECUTADO` no se estimaron. `mod_n`, `mod_dias_ext_sum`, `mod_aprobacion_max_dias` y `dias_adicionados` son cuasi-etiquetas, no features.

<details>
<summary>Código que genera la tabla</summary>

```python
# d se construye con base + columnas de Contratos + agregados de Modificaciones
d["razon_dur"] = np.where(
    d["p_dur_dias"].between(1, 3650) & d["dur_contrato_dias"].between(1, 3650),
    d["dur_contrato_dias"] / d["p_dur_dias"], np.nan,
)
d["razon_valor"] = np.where(
    d["p_precio_base"].gt(0),
    d["valor_del_contrato"] / d["p_precio_base"], np.nan,
)
for var, metodo in [
    ("estado_contrato", "cat"),
    ("fin_dias", "num"),
    ("dur_contrato_dias", "num"),
    ("razon_dur", "num"),
    ("valor_del_contrato", "num"),
    ("razon_valor", "num"),
    ("recursos_suma", "num"),
    ("el_contrato_puede_ser_prorrogado", "cat"),
    ("dias_adicionados", "num"),
    ("mod_n", "num"),
    ("mod_dias_ext_sum", "num"),
    ("mod_aprobacion_max_dias", "num"),
    ("mod_fin_max_dias", "num"),
]:
    f = auc_cat if metodo == "cat" else auc_num
    print(var, f(d, var, "y"), f(d, var, "y_dias"))
```

</details>

### Contraste con las referencias legacy

- `estado_contrato`: 0,820 con `y` y 0,943 con `y_dias`; la referencia es 0,94. Diferencia frente a `y`: 0,120.
- `fecha_de_fin_del_contrato` (0,744), `dur_contrato_dias` (0,740) y `mod_fin_max_dias` (0,747) frente a la referencia 0,71; diferencias 0,034, 0,030 y 0,037.
- `razon_dur`: 0,952 calculado frente a 0,71 de referencia; diferencia 0,242.
- `el_contrato_puede_ser_prorrogado`: 0,563 frente a 0,561 de referencia; diferencia 0,002.
- `recursos_suma`: 0,657 univariado. La referencia legacy 0,751→0,807 corresponde a un modelo con las 39 features, no a esta suma.
- `ultima_actualizacion` (0,74) y el grupo de pagos (0,63) no se pudieron verificar: `NO EJECUTADO`.

**No se ajustó ningún valor de referencia.**

## §9 Verificación de referencias

**Subconjunto:** §9.1 usa todos los contratos 2019–2025, salvo las filas de H6, que usan test 2024. §9.2 usa el subconjunto de cada tabla (todas 2019–2025). §9.3 usa todos los contratos 2019–2025. Las referencias duras son n = 452.494, y = 27,6 % y las tres tasas de H6 en 2024; no se ajustó ningún valor.

### Tabla 9.1. Referencias duras

**Qué muestra:** Contraste de las referencias duras con los valores calculados en esta sesión.

| referencia | esperado | calculado | resultado |
|---|---|---|---|
| n total | 452.494 | 452.494 | coincide |
| y (% sobre 452.494) | 27,6 % | 27,635 % | coincide |
| H6 2024: T3 | 41,6 % | 41,6 % | coincide |
| H6 2024: T1 | 14,3 % | 14,3 % | coincide |
| H6 2024: Sin historia | 11,5 % | 11,5 % | coincide |
| H6 2024: n | 88.546 | 88.546 | coincide |
| H6 2024: tasa base | 24,65 % | 24,65 % | coincide |

**Nota:** no se ajustó ningún valor. La tasa de y se muestra con tres decimales para ver el 27,635 %.

<details>
<summary>Código que genera la tabla</summary>

```python
n_total = len(base)
y_total = 100 * base["y"].mean()
tabla_h6, _ = eda.tasa_entidad_historica(base, anio_objetivo=2024)
n_h6 = int(tabla_h6["n"].sum())
base_h6 = 100 * int(tabla_h6["positivos"].sum()) / n_h6
for g in ["T3", "T1", "Sin historia"]:
    print(g, 100 * tabla_h6.set_index("grupo_hist").loc[g, "tasa"])
print(n_total, y_total, n_h6, base_h6)
```

</details>

### Tabla 9.2. Otras comprobaciones contra el notebook

**Qué muestra:** Contraste de valores intermedios del notebook con las tablas de este documento.

| referencia | esperado | calculado | resultado |
|---|---|---|---|
| H1: tramo 31-60 | 40,03 % | 40,03 % | coincide |
| H3: Contratación directa | 25,23 % | 25,23 % | coincide |
| H4: Último trimestre | 19,33 % | 19,33 % | coincide |
| H5: 366-730 días | 57,10 % | 57,10 % | coincide |
| H7: Obra | 51,85 % | 51,85 % | coincide |
| §7.2: Cédula de Ciudadanía | 27,35 % | 27,35 % | coincide |
| §4.3: Maduro | 27,68 % | 27,68 % | coincide |
| §5: y en 2024 | 24,65 % | 24,65 % | coincide |
| H1: Sin dato | 229 en el texto del notebook | 380 | no coincide: la tabla ejecutada da 380 |

**Nota:** la única diferencia es el texto de H1, que dice 229; `p_fecha_de_publicacion_del` falta en 380 contratos.

<details>
<summary>Código que genera la tabla</summary>

```python
t_h1, i_h1 = tabla_tramos(
    Xy, "dias_publicacion_a_firma",
    bins=[-np.inf, 0, 7, 15, 30, 60, 120, np.inf],
    labels=["0", "1-7", "8-15", "16-30", "31-60", "61-120", ">120"],
)
t_h3, i_h3 = tabla_estandar(
    Xy, "modalidad_de_contratacion", top=12, min_n=500, orden="tasa"
)
Xy_trim = Xy.assign(
    trimestre=Xy["firma_ultimo_trimestre"].map(
        {0: "Resto", 1: "Último trimestre"}
    )
)
t_h4, i_h4 = tabla_estandar(
    Xy_trim, "trimestre", agrupar_pequenas=False, orden="natural",
    orden_natural=["Resto", "Último trimestre"],
)
t_h5, i_h5 = tabla_tramos(
    Xy, "p_dur_dias",
    bins=[-np.inf, 0, 30, 90, 180, 365, 730, np.inf],
    labels=["<=0", "1-30", "31-90", "91-180", "181-365",
            "366-730", ">730"],
)
t_h7, i_h7 = tabla_estandar(
    Xy, "tipo_de_contrato", top=10, min_n=500, orden="tasa"
)
t_doc, i_doc = tabla_estandar(
    base, "tipodocproveedor", min_n=500, orden="tasa"
)
base_madurez = base.assign(
    madurez=base["eval_madura"].map({True: "Maduro", False: "No maduro"})
)
t_mad, i_mad = tabla_estandar(
    base_madurez, "madurez", orden="natural",
    orden_natural=["Maduro", "No maduro"],
)
comp = eda.comparar_etiquetas(base, etiquetas=("y", "y_dias", "y_modif"))
tasas = comp.pivot(index="anio_firma", columns="etiqueta", values="tasa") * 100
for t, col, cat in [
    (i_h1, "dias_publicacion_a_firma", "31-60"),
    (i_h3, "modalidad_de_contratacion", "Contratación directa"),
    (i_h4, "trimestre", "Último trimestre"),
    (i_h5, "p_dur_dias", "366-730"),
    (i_h7, "tipo_de_contrato", "Obra"),
    (i_doc, "tipodocproveedor", "Cédula de Ciudadanía"),
    (i_mad, "madurez", "Maduro"),
]:
    print(cat, t["tabla"].query(f"`{col}` == @cat")["tasa"].iloc[0])
print("y 2024", tasas.loc[2024, "y"])
print("H1 sin dato", int(Xy["dias_publicacion_a_firma"].isna().sum()))
```

</details>

### Tabla 9.3. Referencias legacy de fuga

**Qué muestra:** Contraste de las referencias legacy de ROC-AUC univariado con los valores calculados.

| variable | referencia legacy | AUC vs y | AUC vs y_dias | resultado |
|---|---|---|---|---|
| estado_contrato | 0,94 | 0,820 | 0,943 | no coincide con y; coincide con y_dias |
| fecha_de_fin_del_contrato | 0,71 | 0,744 | — | aproximado (0,744) |
| razon_dur | 0,71 | 0,952 | — | no coincide: 0,952 calculado frente a 0,71 |
| el_contrato_puede_ser_prorrogado | 0,561 | 0,563 | — | coincide (0,563) |
| ultima_actualizacion | 0,74 | NO EJECUTADO | — | no está en data/raw/f2/contratos |
| pagos / facturado / saldos | 0,63 | NO EJECUTADO | — | no descargados |

**Nota:** las referencias legacy no distinguen siempre entre `y` y `y_dias`. `estado_contrato` es el caso claro: 0,94 se reproduce con `y_dias`, no con `y`.

<details>
<summary>Código que genera la tabla</summary>

```python
a_est_y, _ = auc_cat(d, "estado_contrato", "y")
a_est_d, _ = auc_cat(d, "estado_contrato", "y_dias")
a_fin, _ = auc_num(d, "fin_dias", "y")
a_raz, _ = auc_num(d, "razon_dur", "y")
a_pro, _ = auc_cat(d, "el_contrato_puede_ser_prorrogado", "y")
```

</details>

### Diferencias que quedan documentadas

1. `estado_contrato`: 0,94 legacy frente a 0,820 con `y` y 0,943 con `y_dias`. La referencia se reproduce solo con `y_dias`.
2. `razon_dur`: 0,952 calculado frente a 0,71 legacy; diferencia 0,242. `fecha_de_fin_del_contrato` da 0,744 (diferencia 0,034) y `dur_contrato_dias` 0,740 (diferencia 0,030).
3. H1 `Sin dato`: el texto del notebook dice 229; la tabla ejecutada da 380.
4. `fecha_de_fin_del_contrato`: 0,744 frente a 0,71 legacy; diferencia de 0,034, aproximada.
5. `ultima_actualizacion` y el grupo de pagos no se pudieron verificar: `NO EJECUTADO`.

**No se ajustó ningún valor de referencia.**

## §10 Tablas reconstruidas

Tablas que no existían en `notebooks/_borradores/03_eda.ipynb` y se construyeron en esta sesión:

1. §4.3, tabla 4.3.4: madurez × año, con n y tasa de y de maduros y no maduros.
2. §4.3, tabla 4.3.5: composición de los no maduros de 2019–2024 por tipo de contrato.
3. §5, tabla 5.3: 2×2 de `y` y `y_modif`.
4. §5, tabla 5.4: distribución de `y_dias` dentro de `y=0` y `y=1`.
5. §6.4, tabla 6.4.3: calendario mensual octubre 2021–diciembre 2023.
6. §6.4, tabla 6.4.4: calendario mes × modalidad, octubre 2021–diciembre 2023.
7. §6.4, tabla 6.4.5: totales por modalidad en octubre 2021–diciembre 2023.
8. §8, tabla 8.2: ROC-AUC univariado por candidato de fuga, con `y` y `y_dias`.
9. §9, tablas 9.1–9.3: verificación de referencias duras, del notebook y legacy.

Las tablas de §4.3.1–§4.3.3, §5.1–§5.2, §6.1–§6.7 y §7.2 sí existían en el notebook; se reformatearon a las columnas estándar y se agruparon las categorías con n < 500. No se cambiaron los cortes ni las definiciones de las variables.

No se editó `notebooks/_borradores/03_eda.ipynb`.
