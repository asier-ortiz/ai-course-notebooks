# Actividad 7 - HDFS

Este proyecto proporciona una configuración rápida y sencilla para trabajar con **HDFS** usando **Docker**.

---

## Contenido del proyecto

- `Dockerfile` → Construye una imagen basada en `bde2020/hadoop-namenode`, lista para usar HDFS.
- `Makefile` → Automatiza comandos comunes como build, run, stop, logs, etc.
- `.env.example` → Variables de entorno de ejemplo.

---

## Requisitos previos

- [Docker](https://docs.docker.com/get-docker/) instalado
- [Make](https://www.gnu.org/software/make/) instalado (normalmente viene en Linux/Mac, en Windows puede usarse WSL o Git Bash)

---

## Instrucciones de uso

### 1. Clonar el repositorio

```bash
git clone git@github.com:asier-ortiz/ai-course-notebooks.git
cd actividad7
```

### 2. Crear el archivo `.env`

```bash
cp .env.example .env
```

Edita el `.env` si quieres cambiar nombre del contenedor, imagen o ruta del volumen.

### 3. Construir la imagen Docker

```bash
make build
```

### 4. Ejecutar el contenedor

```bash
make run
```

Esto levantará el contenedor con puertos expuestos y volumen montado.

### 5. Acceder al contenedor

Entra en bash dentro del contenedor:

```bash
make bash
```

Desde aquí podrás usar comandos de HDFS, como por ejemplo:

```bash
hdfs dfs -mkdir /books
hdfs dfs -ls /
```

### 6. Otros comandos útiles

- `make stop` → Detener el contenedor
- `make restart` → Reiniciar el contenedor
- `make logs` → Ver logs del contenedor
- `make stats` → Ver estadísticas en vivo
- `make clean` → Eliminar la imagen Docker
- `make prune` → Limpiar redes Docker huérfanas

---

## Puertos expuestos

- **50070** → HDFS NameNode Web UI (acceso desde navegador)
- **9000** → RPC de Hadoop HDFS

Puedes visitar en tu navegador: [http://localhost:50070](http://localhost:50070)

---

## Notas

- Basado en la imagen comunitaria de [Big Data Europe](https://hub.docker.com/r/bde2020/hadoop-namenode).
