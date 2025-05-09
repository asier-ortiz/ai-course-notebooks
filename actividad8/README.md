# Actividad 8 – Apache Kafka con Confluent Cloud

Este proyecto desarrolla un flujo completo de trabajo con Apache Kafka, incluyendo la creación de un topic, un productor de eventos, consumidores individuales y un grupo de consumidores.

La implementación se ha realizado en **Python** utilizando la librería `confluent-kafka`, con conexión directa a un clúster gestionado en **Confluent Cloud**.

---

## Estructura del proyecto

- `.env.example`: plantilla para generar en fichero `.env`.
- `producer.py`: envía mensajes aleatorios al topic creado.
- `consumer.py`: lee los mensajes desde el topic.
- `group_consumer.py`: lanza múltiples instancias de `consumer.py` para simular un grupo de consumidores.
- `utils.py`: carga las variables del `.env`.

---

## Requisitos

Instala las dependencias necesarias:

```bash
pip install confluent-kafka python-dotenv
```

---

## Configuración

Copia el archivo `.env.example` como base:

```bash
cp .env.example .env
```

Luego, edita `.env` y completa las siguientes variables de entorno con los datos proporcionados por Confluent Cloud:

```env
BOOTSTRAP_SERVERS=pkc-xxxxx.us-east-1.aws.confluent.cloud:9092
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here
TOPIC=pec-topic1-asier
```

---

## Ejecución

### Enviar mensajes con el productor:

```bash
python producer.py
```

### Leer mensajes con un único consumidor:

```bash
python consumer.py
```

### Simular un grupo de consumidores (2 procesos):

```bash
python group_consumer.py
```
