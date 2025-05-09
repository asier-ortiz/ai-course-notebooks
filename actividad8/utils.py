import os
from dotenv import load_dotenv


def load_config():
    """
    Carga las variables del archivo .env y devuelve el diccionario
    de configuración necesario para conectar con Confluent Kafka.
    """
    load_dotenv()
    return {
        'bootstrap.servers': os.getenv("BOOTSTRAP_SERVERS"),
        'security.protocol': 'SASL_SSL',
        'sasl.mechanisms': 'PLAIN',
        'sasl.username': os.getenv("API_KEY"),
        'sasl.password': os.getenv("API_SECRET"),
        'group.id': 'python-group',
        'auto.offset.reset': 'earliest'
    }
