"""Artefacto del modelo (contrato 2d): un .joblib con el Pipeline completo y un .json con sus metadatos.

El Pipeline recibe la salida de construir_features. cargar_modelo falla con un mensaje claro si el artefacto no
corresponde a este codigo (features_version, lista de features o version de scikit-learn) o si el archivo cambio.
Solo cargue artefactos de este repositorio: joblib ejecuta codigo al deserializar.
"""
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path

import joblib
import sklearn

from . import features as F


class ModeloIncompatible(RuntimeError):
    """El artefacto no se puede usar con este codigo o entorno."""


def sha256_archivo(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def entorno():
    """Versiones con las que se guarda el modelo (SHAP incluido, porque la app explica con el mismo par)."""
    versiones = {"python": platform.python_version()}
    for paquete in ["scikit-learn", "pandas", "numpy", "shap", "joblib"]:
        try:
            versiones[paquete] = metadata.version(paquete)
        except metadata.PackageNotFoundError:
            versiones[paquete] = None
    return versiones


def guardar_modelo(pipeline, meta, ruta):
    """Guarda ruta (.joblib) y ruta con extension .json. Agrega al json el sha256 del .joblib y devuelve los metadatos."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ruta)
    meta = {**meta, "sha256": sha256_archivo(ruta)}
    ruta.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def cargar_modelo(ruta):
    """Devuelve (pipeline, meta). Lanza ModeloIncompatible con el motivo si algo no coincide."""
    ruta = Path(ruta)
    ruta_meta = ruta.with_suffix(".json")
    for r in (ruta, ruta_meta):
        if not r.exists():
            raise ModeloIncompatible(f"No existe {r}. El artefacto son dos archivos: el .joblib y el .json con el mismo nombre.")
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))

    if meta.get("features_version") != F.FEATURES_VERSION:
        raise ModeloIncompatible(f"El modelo se entreno con features_version={meta.get('features_version')!r} y este codigo "
                                 f"es {F.FEATURES_VERSION!r}: actualice el repositorio o use el modelo que corresponde.")
    if meta.get("features") != F.FEATURES:
        raise ModeloIncompatible("La lista de features del modelo no coincide con src/features.py (FEATURES).")
    version_sk = (meta.get("entorno") or {}).get("scikit-learn")
    if version_sk != sklearn.__version__:
        raise ModeloIncompatible(f"El modelo se guardo con scikit-learn {version_sk} y aqui hay {sklearn.__version__}: "
                                 "instale la version de requirements.txt.")
    if meta.get("sha256") != sha256_archivo(ruta):
        raise ModeloIncompatible(f"El sha256 de {ruta.name} no coincide con el del .json: el archivo cambio o se corrompio.")
    return joblib.load(ruta), meta
