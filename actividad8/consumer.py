#!/usr/bin/env python

from confluent_kafka import Consumer, KafkaError
from utils import load_config
import os

# Cargar configuración y topic
conf = load_config()
topic = os.getenv("TOPIC", "pec-topic1-asier")
consumer_id = os.getenv("CONSUMER_ID", "X")

# Crear instancia del consumidor
consumer = Consumer(conf)
consumer.subscribe([topic])

print(f"[{consumer_id}] Escuchando mensajes en el topic '{topic}'...")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            print(f"[{consumer_id}] Esperando mensajes...")
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                print(f"[{consumer_id}] Error: {msg.error()}")
            continue

        key = msg.key().decode('utf-8') if msg.key() else None
        value = msg.value().decode('utf-8') if msg.value() else None

        print(f"[{consumer_id}] key = {key:10} | value = {value}")

except KeyboardInterrupt:
    print(f"[{consumer_id}] Interrumpido por el usuario.")
finally:
    consumer.close()
