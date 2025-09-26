import os
import streamlit as st

# Archivo de credenciales del servicio (clave de cuenta de servicio de GCP)
SERVICE_ACCOUNT_FILE = "lqairmh-agbg-iberia-song-ccai-04a4f3e29810.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE

# Configuración de BigQuery
BQ_PROJECT = "lqairmh-agbg-iberia-song-ccai"
BQ_DATASET = "ccaiglobalv01"
BQ_TABLE = "ccai-global-transcripts-results-v01"
BQ_LOCATION = "US"  # Ubicación regional

# Nombre completo de la tabla (para insertar en los prompts SQL)
FULL_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"

# Configuración del modelo en Vertex AI desde secrets.toml (más seguro que hardcodear)
VERTEX_PROJECT = st.secrets["vertex"]["project_id"]
VERTEX_LOCATION = st.secrets["vertex"]["location"]
VERTEX_MODEL = st.secrets["vertex"]["model"]

# Alias para consistencia en otros módulos
LOCATION = VERTEX_LOCATION
PROJECT_ID = BQ_PROJECT
