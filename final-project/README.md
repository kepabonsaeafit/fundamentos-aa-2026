# Riesgo de ampliación del plazo en contratos públicos (SECOP II)

Proyecto final de aprendizaje automático (EAFIT, 2026-2). Al firmar un contrato estatal, estima la probabilidad de que termine con el plazo ampliado (prórroga, suspensión o ajuste de fechas), para que veedurías y oficinas de control interno prioricen qué contratos vigilar; ordena por riesgo, no detecta irregularidades. Entrenado con 2019-2023, en el test de 2024 (contratos con plazo vencido) logra PR-AUC 0,584 y ROC-AUC 0,781, frente a 0,470 y 0,682 de la regla «tasa histórica de la entidad» y una tasa base de 0,247.

**Entregable:** [`model_development.ipynb`](model_development.ipynb) (EDA, preprocesamiento, entrenamiento y justificación de las decisiones). **App:** [https://sknotzzxxdi7zwt6652giq.streamlit.app](https://sknotzzxxdi7zwt6652giq.streamlit.app) (Streamlit Community Cloud)

## Fuente de datos

Datos Abiertos Colombia (datos.gov.co), Agencia Nacional de Contratación Pública – Colombia Compra Eficiente. Licencia CC BY-SA 4.0. Conjuntos: Contratos Electrónicos (`jbjy-vk9h`), Procesos de Contratación (`p6dx-8zbt`) y Modificaciones a contratos (`u8cx-r425`).

## Estructura

```
src/             código reutilizable: descarga, base v1, variables, preprocesamiento, modelo, evaluación, SHAP y monitoreo
app/             app de consulta por número de contrato (Streamlit)
docs/            definición del problema, reglas, lista de fuga, plan, despliegue y reajuste
resultados/      salidas de los scripts de src/ que el notebook lee
models/          modelo v1 congelado (.joblib + .json); el dummy es solo para pruebas
tests/           76 pruebas (pytest)
notebooks/       solo _borradores/: borradores de trabajo, superados por model_development.ipynb
00_exploracion/  histórico de la exploración, solo lectura
data/            no se versiona (raw/ inmutable y processed/)
```

## Cómo ejecutarlo

Requiere Python 3.12 o superior (`shap` lo exige). Desde esta carpeta, en CMD de Windows:

```
set PYTHONUTF8=1
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Ejecute el notebook con el kernel del `.venv`.** El kernel `python3` que Jupyter trae por defecto es el Python global, que no tiene `shap`, aunque el `.venv` esté activado: el notebook corre hasta 3.7 y falla en 3.8 con `ModuleNotFoundError: No module named 'shap'`.

- **VS Code:** Select Kernel → Python Environments → `.venv`.
- **Consola:** registre el kernel una sola vez y pásele su nombre a nbconvert:

```
python -m ipykernel install --user --name aa-eda-venv --display-name "AA EDA venv"
jupyter nbconvert --to notebook --execute --inplace model_development.ipynb --ExecutePreprocessor.kernel_name=aa-eda-venv
```

Los datos no se versionan: cómo descargarlos, cuánto tarda cada modo de ejecución (`RECALCULAR`) y el contenido por sección están en la portada del notebook. **Pruebas** (con el entorno activo): `pytest` (76 pruebas, unos 10 s).

**Autores:** Kevin Pabón Niño, Jose Alejandro Jimenez y Geronimo Montes Acebedo.
