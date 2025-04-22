# Actividad 7 - HDFS

Este proyecto proporciona una configuración rápida y sencilla para trabajar con **HDFS** usando **Docker**.

---

## Contenido del proyecto

- `Dockerfile` → Construye una imagen basada en Ubuntu 20.04, instala Hadoop y configura HDFS.
- `Makefile` → Automatiza comandos comunes como build, run, start, stop, logs, etc.
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

Edita el `.env` si quieres cambiar nombre del contenedor, imagen, ruta del volumen o plataforma (`PLATFORM`) en función de tu sistema (`linux/amd64` o `linux/arm64`).

### 3. Construir y arrancar todo automáticamente

```bash
make start
```

Este comando:

- Construirá la imagen Docker
- Lanzará el contenedor en segundo plano
- Formateará automáticamente el NameNode (si no estaba formateado)
- Arrancará los servicios de NameNode y DataNode

¡Todo en un solo paso!

### 4. Acceder al contenedor

Entra en bash dentro del contenedor:

```bash
make bash
```

Desde aquí podrás usar comandos de HDFS, como por ejemplo:

```bash
hdfs dfs -mkdir /books
hdfs dfs -ls /
```

### 5. Otros comandos útiles

- `make stop` → Detener el contenedor
- `make restart` → Reiniciar el contenedor
- `make logs` → Ver logs del contenedor
- `make stats` → Ver estadísticas en vivo del contenedor
- `make clean` → Eliminar la imagen Docker
- `make prune` → Limpiar redes Docker huérfanas

---

## Puertos expuestos

- **50070** → HDFS NameNode Web UI (acceso desde navegador)
- **9000** → RPC de Hadoop HDFS

Puedes visitar en tu navegador: [http://localhost:50070](http://localhost:50070)

---

## Notas

- Basado en instalación manual de Hadoop sobre Ubuntu 20.04.
- Si estás usando un Mac con procesador M1, M2 o M3 (ARM64), asegúrate de configurar correctamente `PLATFORM=linux/amd64` en tu archivo `.env`. Docker se encargará automáticamente de la emulación.

---
