"""App de consulta: riesgo de ampliacion del plazo de un contrato publico (SECOP II).

El usuario pega el numero del contrato (o el enlace de SECOP II). La app consulta datos.gov.co, arma solo las variables
disponibles al firmar (las mismas del entrenamiento) y muestra el nivel de riesgo, el percentil y los factores que mas
pesan, en lenguaje claro.

Uso local (desde final-project/):  streamlit run app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # para importar src/ tambien desde el despliegue

import matplotlib.pyplot as plt
import streamlit as st

from src import api_secop as A
from src import config, explicacion as E, modelo_io

st.set_page_config(page_title="Riesgo de ampliación del plazo", page_icon="📋", layout="centered")

EJEMPLOS = {   # contratos reales de la base de desarrollo, de riesgo alto, medio y bajo con el modelo de prueba
    "Riesgo alto": "CO1.PCCNTR.6169204",
    "Riesgo medio": "CO1.PCCNTR.7136864",
    "Riesgo bajo": "CO1.PCCNTR.6242834",
}
AVISOS = {
    A.AVISO_SIN_PROCESO: "No encontramos el proceso de selección de este contrato en datos.gov.co. Varias variables "
                         "quedaron sin dato y el resultado es menos confiable.",
    A.AVISO_FUERA_DE_PERIODO: f"Este contrato se firmó fuera del periodo con el que se evaluó el modelo "
                              f"({A.ANIO_MIN}-{A.ANIO_MAX}). Tómalo con más cautela.",
    A.AVISO_FECHA_INVALIDA: "La fecha de firma del registro no es válida. El resultado puede ser poco confiable.",
    A.AVISO_MUCHOS_FALTANTES: "Faltan muchos datos de este contrato en SECOP II. El resultado puede ser poco confiable.",
    A.AVISO_DUPLICADO: "El portal devolvió este contrato repetido; se usó el primer registro.",
}
NIVELES = {"alto": ("Riesgo ALTO", st.error), "medio": ("Riesgo MEDIO", st.warning), "bajo": ("Riesgo BAJO", st.success)}


@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_modelo_y_explicador():
    v1 = config.MODELS / "modelo_prorroga_v1.joblib"
    ruta = v1 if v1.exists() else config.MODELS / "modelo_prorroga_dummy.joblib"
    pipeline, meta = modelo_io.cargar_modelo(ruta)
    return pipeline, meta, E.Explicador(pipeline)


@st.cache_resource(show_spinner=False)
def ruta_respaldo():
    """Ruta de la copia local de contratos por si la API no responde (plan C). Puede no existir (p. ej. en la nube).
    Solo se devuelve la ruta: el servicio lee de ahi la fila del contrato pedido y no carga la base en memoria."""
    for ruta in (config.DATA_PROCESSED / "base_contratos.parquet", Path(__file__).parent / "respaldo_demo.parquet"):
        if ruta.exists():
            return ruta
    return None


@st.cache_resource
def cliente():
    return A.crear_cliente(token=A.token_de(st.secrets))     # sin secretos: variable de entorno, .env o sin token


def consultar(texto, explicador, meta):
    """Devuelve (consulta, explicacion|None, nivel, percentil) o muestra el error y devuelve None."""
    try:
        consulta = A.obtener_consulta(texto, cliente=cliente(), respaldo=ruta_respaldo())
    except A.IdInvalido as e:
        st.error(str(e))
    except A.ContratoNoEncontrado as e:
        st.warning(str(e) + " Revisa el número o prueba con el enlace de la página del contrato.")
    except A.ApiNoDisponible as e:
        st.error(str(e))
    else:
        try:
            explicacion = explicador.explicar(consulta.features, consulta.contrato)
            puntaje = explicacion.puntaje
        except E.ExplicacionInvalida:      # no se muestran factores que no sumen bien; el puntaje sigue siendo valido
            explicacion, puntaje = None, float(explicador.pipeline.predict_proba(consulta.features)[0, 1])
        nivel, percentil = E.nivel_y_percentil(puntaje, meta)
        return consulta, explicacion, nivel, percentil
    return None


def frase_percentil(percentil):
    """Percentil en palabras; en los extremos '100 de cada 100' suena raro, asi que se dice 'casi todos'."""
    p = int(round(percentil))
    if p >= 99:
        return "Su puntaje es mayor que el de casi todos los contratos de referencia del modelo."
    if p <= 1:
        return "Su puntaje es menor que el de casi todos los contratos de referencia del modelo."
    return f"Su puntaje es mayor que el de {p} de cada 100 contratos de referencia del modelo."


def tema_oscuro():
    try:
        return st.context.theme.type == "dark"
    except Exception:        # fuera de un navegador (pruebas) o version sin st.context.theme
        return False


def grafico_factores(aportes, oscuro=False):
    """Barras horizontales de los 5 factores principales; solo importa la direccion, no el valor en log-odds."""
    texto = "#fafafa" if oscuro else "#31333f"
    datos = aportes.head(5).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_alpha(0)          # el fondo lo pone la pagina, asi sirve con tema claro y oscuro
    ax.set_facecolor("none")
    ax.barh(datos["nombre"], datos["aporte"], color=["#e74c3c" if a > 0 else "#3498db" for a in datos["aporte"]])
    ax.axvline(0, color=texto, linewidth=0.8)
    ax.set_xlabel("← reduce el riesgo        aumenta el riesgo →", color=texto)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=9, colors=texto)
    for lado in ("top", "right", "bottom"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color(texto)
    fig.tight_layout()
    return fig


def mostrar(consulta, explicacion, nivel, percentil, meta):
    c = consulta.contrato
    st.subheader("El contrato")
    st.markdown(f"**{c['nombre_entidad'] or 'Entidad sin nombre'}**  \n{c['id_contrato']}")
    if c["objeto"]:
        st.caption(c["objeto"][:350] + ("…" if len(c["objeto"]) > 350 else ""))
    detalle = [("Firmado", c["fecha_de_firma"]), ("Modalidad", c["modalidad"]), ("Tipo", c["tipo_de_contrato"]),
               ("Lugar", ", ".join(x for x in (c["ciudad"], c["departamento"]) if x and x != "No Definido") or None)]
    st.markdown("  \n".join(f"{k}: {v}" for k, v in detalle if v))
    if consulta.fuente == "respaldo":
        st.info("La consulta se respondió desde una copia local de los datos, porque datos.gov.co no respondió.")

    st.subheader("Riesgo de ampliación del plazo")
    titulo, caja = NIVELES[nivel]
    caja(f"**{titulo}**. {frase_percentil(percentil)}")

    for clave in consulta.avisos:
        st.warning(AVISOS[clave])

    if explicacion is not None:
        st.subheader("Qué pesa más en este resultado")
        st.caption("Frente a un contrato típico, estas características suben o bajan el riesgo.")
        for r in explicacion.aportes.head(5).itertuples():
            flecha, efecto = ("▲", "aumenta") if r.aporte > 0 else ("▼", "reduce")
            dato = "" if r.valor != E.SIN_DATO else " (dato no registrado)"
            st.markdown(f"{flecha} **{r.nombre}:** {r.valor}{dato}. *{efecto} el riesgo.*")
        st.pyplot(grafico_factores(explicacion.aportes, tema_oscuro()), clear_figure=True)
    else:
        st.info("No se pudo explicar este resultado con la precisión requerida, así que no se muestran los factores.")


def principal():
    st.title("Riesgo de ampliación del plazo de un contrato público")
    st.write("Pega el número de un contrato de **SECOP II** (tiene la forma `CO1.PCCNTR.1234567`) o el enlace de su "
             "página. La herramienta estima qué tan probable es que el plazo del contrato termine ampliándose, "
             "usando solo lo que se sabía al firmarlo.")

    try:
        pipeline, meta, explicador = cargar_modelo_y_explicador()
    except (modelo_io.ModeloIncompatible, E.ModeloNoSoportado) as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        return

    with st.sidebar:
        st.header("Acerca de esta herramienta")
        if str(meta.get("version", "")).startswith("dummy"):
            st.warning("Estás viendo un **modelo de prueba**. Sus resultados no son los del modelo final.")
        st.markdown(f"Modelo: `{meta.get('version')}`")
        st.markdown("Datos: [datos.gov.co](https://www.datos.gov.co), Colombia Compra Eficiente (SECOP II). "
                    "Licencia CC BY-SA 4.0.")
        st.markdown("Código: [repositorio](https://github.com/kepabonsaeafit/fundamentos-aa-2026)")

    def usar_ejemplo(id_contrato):
        st.session_state["entrada"] = id_contrato
        st.session_state["consultar_ya"] = True

    with st.form("consulta"):
        texto = st.text_input("Número o enlace del contrato", key="entrada", placeholder="CO1.PCCNTR.1234567")
        enviar = st.form_submit_button("Consultar riesgo", type="primary")
    st.caption("¿Quieres probar? Elige un ejemplo:")
    columnas = st.columns(len(EJEMPLOS))
    for col, (nombre, idc) in zip(columnas, EJEMPLOS.items()):
        col.button(nombre, key=f"ej_{idc}", on_click=usar_ejemplo, args=(idc,), use_container_width=True)

    if enviar or st.session_state.pop("consultar_ya", False):
        with st.spinner("Consultando datos.gov.co…"):
            resultado = consultar(st.session_state.get("entrada", texto), explicador, meta)
        if resultado:
            mostrar(*resultado, meta)

    st.divider()
    st.caption("Esta herramienta estima el riesgo de que el plazo de un contrato se amplíe. **No detecta irregularidades "
               "ni corrupción**: una ampliación puede ser legítima cuando la causan hechos imprevisibles. Sirve para "
               "decidir qué contratos revisar primero; la decisión de revisar o cuestionar un contrato siempre es de "
               "una persona.")


principal()
