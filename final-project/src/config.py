"""Constantes del proyecto: datasets, semilla, anios del split, rutas y columnas con datos personales."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]      # final-project/
DATA_RAW = RAIZ / "data" / "raw"                # inmutable
DATA_PROCESSED = RAIZ / "data" / "processed"
# Base v1 (452.494 contratos, 2019-2025). En los computadores del equipo data/processed/base_contratos.parquet es la base
# vieja (v0): la v1 esta en la subcarpeta v1_A. Todo lo que necesite la v1 debe leerla de aqui.
BASE_V1 = DATA_PROCESSED / "v1_A" / "base_contratos.parquet"
MODELS = RAIZ / "models"

SEED = 42
BASE_URL = "https://www.datos.gov.co"

# IDs de los conjuntos de datos (datos.gov.co, API Socrata, licencia CC BY-SA 4.0)
DATASETS = {
    "contratos": "jbjy-vk9h",
    "procesos": "p6dx-8zbt",
    "modificaciones": "u8cx-r425",   # mejora de la etiqueta
    "adiciones": "cb9c-h8sn",        # descartado como etiqueta
    "ejecucion": "mfmm-jqmq",        # descartado: es fuga
}

# Split temporal (reglas del proyecto, regla 3)
ANIO_INICIO = 2019      # antes de 2019 dias_adicionados casi no se diligenciaba
ANIO_TEST = 2024
ANIO_DERIVA = 2025      # test de deriva
ANIO_EXCLUIDO = 2026    # censura: la mayoria de los contratos aun no termina

# Columnas con datos personales: nunca se descargan ni se guardan.
# Las 21 primeras son las de 00_exploracion/secop_descarga.py.
PII_EXPLORACION = {
    "nombre_representante_legal", "nacionalidad_representante_legal", "domicilio_representante_legal",
    "tipo_de_identificaci_n_representante_legal", "identificaci_n_representante_legal", "g_nero_representante_legal",
    "nombre_del_banco", "tipo_de_cuenta", "n_mero_de_cuenta",
    "nombre_ordenador_del_gasto", "tipo_de_documento_ordenador_del_gasto", "n_mero_de_documento_ordenador_del_gasto",
    "nombre_supervisor", "tipo_de_documento_supervisor", "n_mero_de_documento_supervisor",
    "nombre_ordenador_de_pago", "tipo_de_documento_ordenador_de_pago", "n_mero_de_documento_ordenador_de_pago",
    "documento_proveedor", "proveedor_adjudicado", "direcci_n_de_ejecuci_n_del_contrato",
}
# Ampliacion aprobada el 2026-10-21: identidad del proveedor y del adjudicador (ver decision de privacidad en el plan)
PII_AMPLIADA = {
    "codigo_proveedor", "nombre_del_adjudicador", "nombre_del_proveedor",
    "nit_del_proveedor_adjudicado", "codigoproveedor", "autor_version",
    # F1 (docs/LISTA_FUGA.md, seccion 7): puede ser un ID de usuario / el texto a veces trae nombres de personas
    "sustituto_version", "proposito_modificacion",
}
PII = PII_EXPLORACION | PII_AMPLIADA
