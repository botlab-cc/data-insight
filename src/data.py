from google.cloud import bigquery
from google.oauth2 import service_account
from src.config import SERVICE_ACCOUNT_FILE, BQ_PROJECT, BQ_LOCATION

# Cliente de BigQuery con credenciales explícitas
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
client = bigquery.Client(credentials=credentials, project=BQ_PROJECT, location=BQ_LOCATION)

def run_query(sql: str):
    """
    Ejecuta una consulta SQL en BigQuery y devuelve un DataFrame con los resultados.
    """
    job = client.query(sql, location=BQ_LOCATION)
    return job.result().to_dataframe()
