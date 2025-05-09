#!/usr/bin/env python

from random import choice
from confluent_kafka import Producer
from utils import load_config
import os

# Cargar configuración desde .env
conf = load_config()
topic = os.getenv("TOPIC", "pec-topic1-asier")

# Crear instancia del productor
producer = Producer(conf)


# Callback para confirmar entregas
def delivery_callback(err, msg):
    if err:
        print(f"[ERROR] Fallo al entregar mensaje: {err}")
    else:
        print(f"[OK] Enviado a {msg.topic()}: key = {msg.key().decode('utf-8')}, value = {msg.value().decode('utf-8')}")


# Listas de ejemplo
user_ids = ['eabara', 'jsmith', 'sgarcia', 'jbernard', 'htanaka', 'awalther']
products = ['book', 'alarm clock', 't-shirts', 'gift card', 'batteries']

# Enviar 10 mensajes aleatorios
for _ in range(10):
    user_id = choice(user_ids)
    product = choice(products)
    producer.produce(topic, key=user_id, value=product, callback=delivery_callback)
    producer.poll(0)  # Ejecuta los callbacks pendientes

producer.flush()  # Espera hasta que todos los mensajes hayan sido entregados
