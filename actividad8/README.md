# Actividad 8 - Kafka + Zookeeper

Este proyecto proporciona una configuración rápida y sencilla para trabajar con **Apache Kafka** y **Zookeeper** usando **Docker Compose**.

---

## Contenido del proyecto

- `docker-compose.yml` → Orquesta contenedores de Zookeeper y Kafka.
- `Makefile` → Automatiza comandos comunes como build, up, down, logs, etc.
- `.env.example` → Variables de entorno de ejemplo para personalizar puertos, versiones e imagenes.

---

## Requisitos previos

- [Docker](https://docs.docker.com/get-docker/) instalado
- [Docker Compose](https://docs.docker.com/compose/) instalado
- [Make](https://www.gnu.org/software/make/) instalado

---

## Instrucciones de uso

### 1. Clonar el repositorio

```bash
git clone git@github.com:asier-ortiz/ai-course-notebooks.git
cd actividad8
```

### 2. Crear el archivo `.env`

```bash
cp .env.example .env
```

Edita el `.env` si quieres cambiar versiones de Kafka, Zookeeper, puertos o nombres de contenedores.

### 3. Levantar los servicios

```bash
make up
```

Esto lanzará tanto **Zookeeper** como **Kafka** automáticamente en segundo plano.

### 4. Acceder a los contenedores

Para entrar en el contenedor de Kafka:

```bash
make bash-kafka
```

Para entrar en el contenedor de Zookeeper:

```bash
make bash-zookeeper
```

### 5. Otros comandos útiles

- `make down` → Parar y eliminar los contenedores
- `make restart` → Reiniciar los servicios
- `make logs` → Ver logs de los contenedores
- `make clean` → Borrar todos los contenedores e imágenes relacionados
- `make prune` → Limpiar redes Docker huérfanas

---

## Puertos expuestos

- **2181** → Zookeeper client port
- **9092** → Kafka broker port

Puedes conectar clientes Kafka apuntando a `localhost:9092`.

---

## Notas

- Se utiliza `confluentinc/cp-zookeeper` y `confluentinc/cp-kafka`, imágenes oficiales de Confluent Platform.
- Proyecto preparado para funcionar en arquitecturas `linux/amd64` y `linux/arm64`.
