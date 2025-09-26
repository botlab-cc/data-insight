from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.cloud import bigquery
from google.oauth2 import service_account

from src.config import (
    SERVICE_ACCOUNT_FILE,
    VERTEX_MODEL,
    VERTEX_LOCATION,
    BQ_PROJECT,
    BQ_DATASET,
    BQ_TABLE,
)

# Inicializar cliente BigQuery
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
bq_client = bigquery.Client(credentials=credentials, project=BQ_PROJECT, location=VERTEX_LOCATION)

# Modelo de Vertex AI
MODEL = VERTEX_MODEL
TABLE_ID = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"


def load_table_schema() -> str:
    """
    Carga el esquema de la tabla para usarlo en el prompt SQL.
    """
    table = bq_client.get_table(TABLE_ID)
    schema_lines = [f"- {field.name} ({field.field_type})" for field in table.schema]
    return "\n".join(schema_lines)


def build_prompt_sql(user_input: str) -> str:
    """
    Construye el prompt completo para generar SQL en BigQuery.
    """
    schema = load_table_schema()
    return f"""
Actúa como un experto en análisis de datos de Contact Center.

Tu tarea es generar una consulta SQL de BigQuery a partir de la siguiente tabla:
{TABLE_ID}

Esquema de la tabla:
{schema}

Pregunta del usuario:
{user_input}

Reglas obligatorias:
- Devuelve únicamente una consulta SQL 100% válida y ejecutable.
- No incluyas explicaciones, markdown, encabezados ni comentarios.
- Usa comillas simples ('') para valores string.
- Aplica LOWER() para comparaciones con strings.
- Si hay ambigüedad, usa 'No Aplica' como valor por defecto.
- Usa funciones estándar de BigQuery (como `EXTRACT()`, `DATE_SUB()`, etc).
- Si comparas campos tipo TIMESTAMP con fechas relativas (como último mes), convierte el campo con DATE(campo) y usa DATE_SUB(CURRENT_DATE(), INTERVAL ...).
- Finaliza siempre la consulta con punto y coma.
""".strip()


def generate_sql(user_prompt: str) -> str:
    """
    Genera una consulta SQL a partir de un prompt en lenguaje natural.
    """
    try:
        model = GenerativeModel(MODEL)
        chat = model.start_chat()

        prompt = build_prompt_sql(user_prompt)

        response = chat.send_message(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.2,
                top_p=0.8,
                top_k=40,
                max_output_tokens=1024,
            )
        )
        text = response.text.strip()

        # Validaciones básicas
        if not text.lower().startswith("select"):
            raise RuntimeError(f"SQL inválido: {text}")
        if "from" not in text.lower():
            raise RuntimeError(f"SQL incompleta (falta FROM): {text}")
        if ";" in text:
            text = text.split(";")[0].strip()

        return text

    except Exception as e:
        raise RuntimeError(f"Error generando SQL: {e}")


def generate_recommendation(user_prompt: str, context_text: str) -> str:
    """
    Genera una recomendación de mejora basada en la consulta y los resultados.
    """
    prompt = f"""
Actúa como analista de datos de un Contact Center.

Basándote en la siguiente pregunta del usuario y en los datos obtenidos,
redacta una recomendación clara y directa para mejorar la experiencia del cliente
o la eficiencia operativa. Siempre debes dar una recomendación, incluso si los datos son pocos.

Pregunta:
{user_prompt}

Datos:
{context_text}

Recomendación:
""".strip()

    try:
        model = GenerativeModel(MODEL)
        chat = model.start_chat(response_validation=False)

        response = chat.send_message(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.5,
                max_output_tokens=2048,
            )
        )

        text = response.text.strip()
        if not text:
            return "Con base en los datos, se recomienda revisar tendencias adicionales para mejorar la atención al cliente."
        return text

    except Exception as e:
        return f"No se pudo generar una recomendación clara: {e}"
