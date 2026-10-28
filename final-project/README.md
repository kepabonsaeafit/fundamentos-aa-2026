# Riesgo de ampliación del plazo en contratos públicos (SECOP II)

Proyecto final de aprendizaje automático (EAFIT, 2026-2). Modelo que, al firmar un contrato estatal, estima la probabilidad de que termine con el plazo ampliado. El entregable es [`model_development.ipynb`](model_development.ipynb): EDA, preprocesamiento, entrenamiento y justificación de las decisiones.

## Entorno

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

Los datos no se versionan. Cómo obtenerlos, cuánto tarda cada modo de ejecución (`RECALCULAR`) y el contenido por sección están en la portada del notebook.
