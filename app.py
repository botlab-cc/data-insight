import streamlit as st
from src.llm import generate_sql, generate_recommendation
from src.data import run_query
from src.config import PROJECT_ID, BQ_TABLE, BQ_DATASET, BQ_PROJECT
from src.utils import dataframe_to_text
import base64

st.set_page_config(
    page_title="Quality Assistant",
    layout="wide",
    page_icon="assets/logo.png"
)

if "history" not in st.session_state:
    st.session_state.history = []

debug = st.sidebar.checkbox("Debug", value=False)

def img_to_base64(path):
    with open(path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    return b64

b64_logo = img_to_base64("assets/logo.png")

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 1rem;">
        <img src="data:image/png;base64,{b64_logo}" alt="Logo" style="height: 50px;">
        <h1 style="color:#7300A0; margin: 0;">Quality Assistant</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("¿Qué necesitas saber? (ej. % de llamadas con frustración en Energía)")

with st.form(key="query_form", clear_on_submit=False):
    user_input = st.text_input(
        "",
        value="¿Cuál fue el total de llamadas en el último mes?",
        placeholder="Escribe tu pregunta aquí...",
        key="user_input_key",
    )
    submit_button = st.form_submit_button("Enviar Consulta")

if submit_button and user_input.strip():
    try:
        with open("prompts/prompt_sql.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        st.error("Archivo prompts/prompt_sql.txt no encontrado.")
        st.stop()

    prompt_sql = prompt_template.format(
        project_id=f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}",
        user_input=user_input.strip()
    )

    with st.spinner("Generando consulta SQL con Gemini..."):
        sql_query = generate_sql(prompt_sql)

    if debug:
        st.subheader("SQL generado")
        st.code(sql_query, language="sql")

    if len(sql_query) < 10:
        st.error("Consulta SQL vacía o demasiado corta. No se ejecutará.")
    else:
        try:
            df = run_query(sql_query)
            if df.empty:
                st.warning("La consulta no devolvió resultados.")
            else:
                st.dataframe(df)

                df_text = dataframe_to_text(df)

                with st.spinner("Generando recomendación de mejora..."):
                    recommendation = generate_recommendation(user_input, df_text)

                st.markdown("### Recomendación de Mejora")
                st.markdown(
                    f"""
                    <div class="recommendation-box">{recommendation}</div>
                    """,
                    unsafe_allow_html=True
                )

            st.session_state.history.append({"pregunta": user_input, "sql": sql_query})
        except Exception as e:
            st.error(f"Error al ejecutar la consulta: {e}")

with st.sidebar:
    with st.expander("Preguntas Frecuentes", expanded=True):
        try:
            with open("assets/faqs.txt", "r", encoding="utf-8") as f:
                faqs = [line.strip() for line in f if line.strip()]
            for faq in faqs:
                st.markdown(f"- {faq}")
        except FileNotFoundError:
            st.warning("Archivo assets/faqs.txt no encontrado.")

    with st.expander("Historial de consultas", expanded=True):
        if st.session_state.history:
            for i, item in enumerate(reversed(st.session_state.history[-20:])):
                st.markdown(f"**Pregunta:** {item['pregunta']}")
                if debug:
                    st.code(item["sql"], language="sql")
                if i < len(st.session_state.history[-20:]) - 1:
                    st.markdown("---")
        else:
            st.write("No hay consultas en el historial.")
