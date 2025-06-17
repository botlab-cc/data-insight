import streamlit as st
import os
import json
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery, aiplatform

# Configuración del entorno GCP
PROJECT_ID = "lqairmh-agbg-iberia-song-ccai"
LOCATION = "us-central1"
TABLE_FULL = "lqairmh-agbg-iberia-song-ccai.ccaiglobalv01.ccai-global-transcripts-results-v01"

# Autenticación
credentials_info = json.loads(st.secrets["gcp_service_account"])
credentials = service_account.Credentials.from_service_account_info(credentials_info)

client_bq = bigquery.Client(credentials=credentials, project=PROJECT_ID)
aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
chat_model = aiplatform.ChatModel.from_pretrained("chat-bison@001")

# Prompt optimizado con conocimiento del esquema
def nl_to_sql(question: str) -> str:
    prompt = f"""
Eres un experto en análisis de datos. Tu tarea es traducir preguntas en lenguaje natural
a consultas SQL sobre la siguiente tabla de BigQuery:

Tabla: `{TABLE_FULL}`

Campos disponibles:
- motivo_llamada_principal, motivo_llamada_secundario, motivo_llamada_terciario
- frustracion_cliente, recomendacion_agente_l1, recomendacion_agente_l2
- solucion_recomendacion, posibilidad_baja_producto, producto
- fcr, nps, csat, tasa_transferencia, tasa_abandono, churn
- customer_engagement, resumen_llamada, fecha_llamada, cliente, geografia, tramo_edad, antiguedad, tipo_agente

Reglas:
- Siempre usa la tabla `{TABLE_FULL}`
- La salida debe ser solo una consulta SQL válida de BigQuery, sin explicaciones ni comentarios
- Usa funciones como COUNT, AVG, GROUP BY, ORDER BY según la pregunta
- Si se pide filtrar por fechas, usa la columna `fecha_llamada`

Pregunta: {question}
SQL:
"""
    response = chat_model.predict(prompt=prompt, temperature=0)
    return response.text.strip()

def run_query(sql: str) -> pd.DataFrame:
    query_job = client_bq.query(sql)
    return query_job.result().to_dataframe()

# Interfaz tipo chat
st.title("💬 Chat con datos de Contact Center")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_prompt = st.chat_input("Haz una pregunta sobre tus datos...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            try:
                sql = nl_to_sql(user_prompt)
                st.code(sql, language="sql")

                df = run_query(sql)
                if df.empty:
                    result_text = "✅ Consulta ejecutada, pero no hay resultados."
                else:
                    result_text = f"🟢 Resultado:\n\n{df.to_markdown(index=False)}"

                st.session_state.messages.append({"role": "assistant", "content": result_text})
                st.markdown(result_text)
            except Exception as e:
                error_msg = f"❌ Error: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.error(error_msg)

