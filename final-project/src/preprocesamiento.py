"""Preprocesamiento (J2, duenio: Jose): un ColumnTransformer por familia de modelo, dentro del Pipeline.

Principio del plan (seccion 2): src/features.py hace lo determinista fila a fila; aqui va todo lo que se aprende de
los datos (cuantiles de la winsorizacion, imputacion, codificacion, escalado, target encoding). El Pipeline se ajusta
solo con train, asi que no hay fuga, y la app reproduce exactamente el entrenamiento. No se modifica features.py.

Familias:
- "arbol"      HGB, arbol de decision y RF. Categoricas -> codigos ordinales (top 200; desconocidas y faltantes -> NaN),
               nit_entidad -> TargetEncoder, numericas crudas (los arboles manejan NaN). SIN categorias nativas: con
               ellas TreeExplainer da atribuciones incorrectas y la app se niega a explicar (plan, seccion 0 punto 7).
- "arbol_nativo"  Igual, pero para comparar contra un HGB con categorical_features="from_dtype". Solo para medir en J3;
               no se usa en el artefacto.
- "distancia"  kNN y SVM. One-hot (categorias < 1 % agrupadas), nit_entidad -> TargetEncoder, numericas con log1p
               en los conteos, mes y dia de la semana ciclicos, imputacion por mediana con indicador, RobustScaler.

Comun a todas:
- p_dur_dias: 0 -> faltante + indicador p_dur_dias_cero (5,3 % de los maduros; tasa de y distinta, 0,247 frente a
  0,279) y winsorizacion por arriba al cuantil 99,9 % del train (hay una duracion de 4.078 millones de dias en la v1).
- p_proveedores_que_manifestaron se descarta: es constante 0 en el train.
- Toda columna de salida se llama como su feature o "feature_algo", para que la app agregue los valores SHAP por
  feature original (src/explicacion.py, _original_de).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, TargetEncoder
from sklearn.utils.validation import check_is_fitted

from . import config
from . import features as F

FAMILIAS = ("arbol", "arbol_nativo", "distancia")
TOP_CATEGORIAS = 200          # codigos ordinales: el resto de categorias comparte un codigo (como la F0 y el dummy)
MIN_FRECUENCIA_ONEHOT = 0.01  # one-hot: categorias con < 1 % del train se agrupan en "infrequent"
CUANTIL_DURACION = 0.999      # winsorizacion de p_dur_dias (aprendida del train)

ENTIDAD = ["nit_entidad"]
CONSTANTES = ["p_proveedores_que_manifestaron"]
DURACION = ["p_dur_dias"]
# Pares casi identicos (Spearman >= 0,995 en el EDA): en distancia contarian doble, se deja un representante.
REDUNDANTES_DISTANCIA = ["p_proveedores_con_invitacion", "p_proveedores_unicos_con"]
CONTEOS = ["p_numero_de_lotes", "p_proveedores_invitados", "p_proveedores_con_invitacion",
           "p_respuestas_al_procedimiento", "p_respuestas_externas", "p_conteo_de_respuestas_a_ofertas",
           "p_proveedores_unicos_con"]
CICLICAS = {"mes_firma": 12, "dia_semana": 7}
BINARIAS = ["firma_ultimo_trimestre", "fin_planeado_cruza_anio"]


# ---------------------------------------------------------------- transformadores propios
class DuracionLimpia(BaseEstimator, TransformerMixin):
    """p_dur_dias: 0 o negativa -> NaN (+ indicador <col>_cero) y recorte superior al cuantil del train."""

    def __init__(self, cuantil=CUANTIL_DURACION, indicador=True):
        self.cuantil = cuantil
        self.indicador = indicador

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        v = X.where(X > 0)
        self.tope_ = v.quantile(self.cuantil).to_numpy(dtype="float64")
        return self

    def transform(self, X):
        check_is_fitted(self, "tope_")
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        cero = X <= 0
        out = X.where(~cero).clip(upper=pd.Series(self.tope_, index=X.columns), axis=1).astype("float64")
        if self.indicador:
            out = pd.concat([out, cero.astype("float64").add_suffix("_cero")], axis=1)
        return out

    def get_feature_names_out(self, input_features=None):
        nombres = list(self.feature_names_in_)
        return np.asarray(nombres + ([f"{c}_cero" for c in nombres] if self.indicador else []), dtype=object)


class Ciclica(BaseEstimator, TransformerMixin):
    """Mes y dia de la semana como seno y coseno (en distancia, diciembre queda cerca de enero). NaN -> 0, 0."""

    def __init__(self, periodos=None):
        self.periodos = periodos

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(pd.DataFrame(X).columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        out = {}
        for c in X.columns:
            ang = 2 * np.pi * X[c].astype("float64") / (self.periodos or CICLICAS)[c]
            out[f"{c}_sin"], out[f"{c}_cos"] = np.sin(ang).fillna(0), np.cos(ang).fillna(0)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_{t}" for c in self.feature_names_in_ for t in ("sin", "cos")], dtype=object)


class Log1p(BaseEstimator, TransformerMixin):
    """log1p de conteos no negativos (los ceros son 'sin respuesta', no faltantes)."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(pd.DataFrame(X).columns, dtype=object)
        return self

    def transform(self, X):
        return np.log1p(pd.DataFrame(X, columns=self.feature_names_in_).clip(lower=0))

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class ImputarConIndicador(BaseEstimator, TransformerMixin):
    """Mediana del train + columna <col>_faltante solo para las columnas con faltantes en el train."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.mediana_ = X.median().fillna(0)
        self.con_faltantes_ = [c for c in X.columns if X[c].isna().any()]
        return self

    def transform(self, X):
        check_is_fitted(self, "mediana_")
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        ind = X[self.con_faltantes_].isna().astype("float64").add_suffix("_faltante")
        return pd.concat([X.fillna(self.mediana_), ind], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + [f"{c}_faltante" for c in self.con_faltantes_], dtype=object)


# ---------------------------------------------------------------- preprocesadores
def _columnas(excluir):
    excluir = set(excluir) | set(CONSTANTES)
    desconocidas = excluir - set(F.FEATURES)
    if desconocidas:
        raise ValueError(f"excluir nombra columnas que no son FEATURES: {sorted(desconocidas)}")
    cat = [c for c in F.FEATURES_CAT if c not in excluir and c not in ENTIDAD]
    num = [c for c in F.FEATURES_NUM if c not in excluir]
    ent = [c for c in ENTIDAD if c not in excluir]
    return cat, num, ent


def _target_encoder():
    # cv interno: en fit_transform cada fila se codifica con los otros pliegues (sin fuga dentro del train);
    # entidades no vistas -> media global del train.
    return TargetEncoder(target_type="binary", smooth="auto",
                         cv=StratifiedKFold(5, shuffle=True, random_state=config.SEED))


def construir_preprocesador(familia="arbol", excluir=()):
    """ColumnTransformer para la familia dada. `excluir`: features que se dejan fuera (ablacion de J3)."""
    if familia not in FAMILIAS:
        raise ValueError(f"familia debe ser una de {FAMILIAS}")
    cat, num, ent = _columnas(excluir)
    dur = [c for c in DURACION if c in num]
    pasos = []

    if familia in ("arbol", "arbol_nativo"):
        codigos = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                 encoded_missing_value=np.nan, max_categories=TOP_CATEGORIAS)
        if familia == "arbol_nativo":
            # salida 'category' para que HGB(categorical_features="from_dtype") las trate como nativas
            codigos = Pipeline([("cod", codigos), ("a_categoria", _ACategoria())])
        pasos += [("cat", codigos, cat), ("dur", DuracionLimpia(), dur),
                  ("num", "passthrough", [c for c in num if c not in dur])]
    else:
        num = [c for c in num if c not in REDUNDANTES_DISTANCIA]
        ciclicas = [c for c in num if c in CICLICAS]
        conteos = [c for c in num if c in CONTEOS]
        binarias = [c for c in num if c in BINARIAS]
        resto = [c for c in num if c not in ciclicas + conteos + binarias + dur]
        onehot = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="SIN_DATO")),
                           ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_FRECUENCIA_ONEHOT,
                                                sparse_output=False))])
        escalar = lambda *previos: Pipeline([*previos, ("imp", ImputarConIndicador()), ("esc", RobustScaler())])
        pasos += [("cat", onehot, cat),
                  ("dur", escalar(("limpia", DuracionLimpia()), ("log", Log1p())), dur),
                  ("conteos", escalar(("log", Log1p())), conteos),
                  ("num", escalar(), resto),
                  ("ciclicas", Ciclica(), ciclicas),
                  ("binarias", ImputarConIndicador(), binarias)]
    pasos.append(("entidad", _target_encoder(), ent))
    pasos = [p for p in pasos if len(p[2])]
    ct = ColumnTransformer(pasos, remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


class _ACategoria(BaseEstimator, TransformerMixin):
    """Codigos ordinales -> dtype 'category' (solo para la variante nativa de comparacion)."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(pd.DataFrame(X).columns, dtype=object)
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names_in_).astype("category")

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def pipeline(estimador, familia="arbol", excluir=()):
    """Pipeline([("prep", preprocesador), ("modelo", estimador)]). Recibe la salida de construir_features."""
    return Pipeline([("prep", construir_preprocesador(familia, excluir)), ("modelo", estimador)])


def pipeline_hgb_defecto(excluir=()):
    """HGB por defecto (sin early stopping, regla 4) sobre la familia arbol. Es el que usa G3 mientras llega J3."""
    return pipeline(HistGradientBoostingClassifier(early_stopping=False, random_state=config.SEED), "arbol", excluir)
