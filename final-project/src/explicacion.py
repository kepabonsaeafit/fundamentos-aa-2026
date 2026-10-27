"""Explicacion de un puntaje para la app (G1): nivel de riesgo, percentil y factores en lenguaje claro.

SHAP: TreeExplainer sobre el estimador final del Pipeline, con pipeline[:-1].transform(X). Las columnas transformadas se
agregan a la feature original (si el preprocesamiento usa one-hot, varias columnas suman en una) y se muestran con
F.NOMBRES_LEGIBLES.

Comprobacion obligatoria (plan, seccion 0 punto 7): TreeExplainer sobre un HistGradientBoostingClassifier con categorias
nativas NO lanza error pero da atribuciones incorrectas. Por eso:
- Si el estimador tiene categorias nativas, el Explicador se niega a crearse (ModeloNoSoportado): el equipo decide entre
  (a) un explicador generico sobre predict_proba o (b) quitar las nativas.
- En cada explicacion se verifica la aditividad: suma de los valores SHAP + valor base = salida del modelo en log-odds,
  con error maximo < 1e-3. Si no se cumple, se lanza ExplicacionInvalida y la app muestra el puntaje sin factores.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from . import features as F

TOLERANCIA_ADITIVIDAD = 1e-3
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
SIN_DATO = "sin dato"


class ModeloNoSoportado(NotImplementedError):
    """El estimador final no se puede explicar de forma exacta con TreeExplainer."""


class ExplicacionInvalida(RuntimeError):
    """La suma de los valores SHAP no reproduce la salida del modelo: no se deben mostrar los factores."""


def nivel_y_percentil(puntaje, meta):
    """Nivel (bajo, medio, alto) y percentil del puntaje frente a la referencia guardada con el modelo.

    La referencia (percentiles_puntaje, 101 cuantiles) y los umbrales salen del .json del artefacto. La app muestra
    estos dos datos y no la probabilidad cruda, que con la deriva queda descalibrada.
    """
    umbrales = meta["umbrales_riesgo"]
    nivel = "alto" if puntaje >= umbrales["alto"] else "medio" if puntaje >= umbrales["medio"] else "bajo"
    cuantiles = meta["percentiles_puntaje"]
    percentil = float(np.interp(puntaje, cuantiles, np.linspace(0, 100, len(cuantiles))))
    return nivel, percentil


# ---------------------------------------------------------------- valores en lenguaje claro
def _numero_es(x, decimales=0):
    """12345.6 -> '12.346' (punto de miles, coma decimal)."""
    texto = f"{x:,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def _pesos(valor):
    v = float(valor)
    if v >= 1e9:
        return f"$ {_numero_es(v / 1e9, 1)} mil millones"
    if v >= 1e6:
        return f"$ {_numero_es(v / 1e6, 1)} millones"
    return f"$ {_numero_es(v)}"


def valor_legible(feature, valor, contrato=None):
    """Valor de una feature como se lo explicaria a una persona (la feature numerica puede ser un logaritmo o un codigo)."""
    if valor is None or (not isinstance(valor, str) and pd.isna(valor)):
        return SIN_DATO
    if feature == "nit_entidad":
        return (contrato or {}).get("nombre_entidad") or f"NIT {valor}"
    if feature in ("log_precio_base", "log_pago_adelantado"):
        return _pesos(np.expm1(float(valor)))
    if feature == "len_objeto":
        return f"{int(valor)} caracteres"
    if feature in ("dias_firma_a_inicio", "dias_publicacion_a_firma", "p_dur_dias"):
        d = int(round(float(valor)))
        return f"{_numero_es(d)} día" + ("" if abs(d) == 1 else "s")
    if feature == "mes_firma":
        return MESES[int(valor) - 1]
    if feature == "dia_semana":
        return DIAS[int(valor)]
    if feature in ("firma_ultimo_trimestre", "fin_planeado_cruza_anio"):
        return "Sí" if float(valor) >= 0.5 else "No"
    if feature == "unspsc_segmento":
        return f"segmento {valor}"
    if feature == "unspsc_familia":
        return f"familia {valor}"
    if isinstance(valor, (int, float, np.integer, np.floating)):
        return _numero_es(float(valor), 0 if float(valor) == int(valor) else 2)
    return str(valor)


# ---------------------------------------------------------------- SHAP
def _original_de(nombre, features):
    """Feature original de una columna transformada (la misma, o 'prefijo__feature' o 'feature_categoria' en one-hot)."""
    nombre = nombre.split("__")[-1]
    candidatas = [f for f in features if nombre == f or nombre.startswith(f + "_")]
    if not candidatas:
        raise ValueError(f"la columna transformada {nombre!r} no corresponde a ninguna feature")
    return max(candidatas, key=len)


@dataclass
class Explicacion:
    puntaje: float              # probabilidad cruda: solo para ordenar y depurar, la app no la muestra
    aportes: pd.DataFrame       # una fila por feature: feature, nombre, valor, aporte (log-odds), orden por |aporte|
    error_aditividad: float


class Explicador:
    """TreeExplainer sobre el Pipeline guardado. Se crea una vez (en la app, con st.cache_resource)."""

    def __init__(self, pipeline, features=None):
        self.pipeline = pipeline
        self.prep, self.modelo = pipeline[:-1], pipeline[-1]
        self.features = list(features or F.FEATURES)
        nativas = getattr(self.modelo, "is_categorical_", None)
        if nativas is not None and np.any(nativas):
            raise ModeloNoSoportado(
                "El modelo usa categorias nativas: TreeExplainer daria atribuciones incorrectas sin avisar "
                "(plan, seccion 0 punto 7). Hay que usar un explicador generico o entrenar sin categorias nativas.")
        self.explicador = shap.TreeExplainer(self.modelo)
        self.base = float(np.ravel(self.explicador.expected_value)[0])

    def explicar(self, X, contrato=None):
        """Explicacion de la primera fila de X (la salida de construir_features) con la comprobacion de aditividad."""
        X = X.iloc[[0]]
        Xt = self.prep.transform(X)
        valores = self.explicador.shap_values(Xt)
        valores = valores[1] if isinstance(valores, list) else valores
        salida = float(self.modelo.decision_function(Xt)[0])
        error = abs(float(valores[0].sum()) + self.base - salida)
        if error >= TOLERANCIA_ADITIVIDAD:
            raise ExplicacionInvalida(f"la suma de los valores SHAP no reproduce la salida del modelo "
                                      f"(error {error:.2e} >= {TOLERANCIA_ADITIVIDAD:.0e})")
        nombres = list(self.prep.get_feature_names_out())
        originales = [_original_de(n, self.features) for n in nombres]
        por_feature = pd.Series(valores[0], index=originales).groupby(level=0).sum()
        tabla = pd.DataFrame({
            "feature": por_feature.index,
            "nombre": [F.NOMBRES_LEGIBLES[f] for f in por_feature.index],
            "valor": [valor_legible(f, X.iloc[0][f], contrato) for f in por_feature.index],
            "aporte": por_feature.to_numpy(),
        })
        tabla = tabla.reindex(tabla["aporte"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        puntaje = float(self.pipeline.predict_proba(X)[0, 1])
        return Explicacion(puntaje=puntaje, aportes=tabla, error_aditividad=error)
