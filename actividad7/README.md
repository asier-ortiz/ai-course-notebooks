# Actividad 7 - HDFS

Este proyecto proporciona una configuración rápida y automatizada para trabajar con **HDFS** usando **Docker**, cargando automáticamente contenido al sistema distribuido al iniciar el contenedor.

---

## Estructura y contenido del proyecto

```
actividad7/
├── Dockerfile
├── Makefile
├── .env.example
├── entrypoint.sh
├── setup/
│   └── actividad7.sh
├── hadoop-config/
│   ├── core-site.xml
│   └── hdfs-site.xml
└── home/
    └── ubuntu/
        └── bigdata/
            └── examples/
                ├── books/
                │   ├── frankenstein.txt
                │   └── Quijote.txt
                └── hdfs/
                    └── fichero_result.zip
```

- `Dockerfile` → Imagen personalizada basada en Ubuntu 20.04 con Hadoop 2.7.4 y configuración manual de HDFS.
- `Makefile` → Automatiza tareas como build, run, stop, logs, stats, etc.
- `.env.example` → Archivo de ejemplo con variables de entorno configurables.
- `entrypoint.sh` → Script de arranque que inicia los servicios y lanza `actividad7.sh`.
- `setup/actividad7.sh` → Script que realiza todas las tareas del ejercicio: crear directorios, cargar ficheros, descomprimir y mover archivos en HDFS.
- `hadoop-config/` → Archivos XML con configuración de HDFS.
- `home/ubuntu/bigdata/examples/` → Estructura de carpetas que simula un entorno Linux típico dentro del contenedor. Aquí deben colocarse los archivos de entrada para HDFS:
  - `/books` → Ficheros de texto de ejemplo.
  - `/hdfs` → Archivo comprimido `fichero_result.zip`.

---

## Requisitos previos

- [Docker](https://docs.docker.com/get-docker/)
- [Make](https://www.gnu.org/software/make/) (en Linux/Mac viene por defecto; en Windows puedes usar WSL o Git Bash)

---

## Instrucciones de uso

### 1. Clonar el repositorio

```bash
git clone git@github.com:asier-ortiz/ai-course-notebooks.git
cd ai-course-notebooks/actividad7
```

### 2. Crear el archivo `.env`

```bash
cp .env.example .env
```

Edita el `.env` si quieres cambiar:

- Nombre de imagen o contenedor
- Plataforma (`PLATFORM`: `linux/amd64` o `linux/arm64`)
- Ruta del volumen de entrada (`INPUT_DIR`)
- Usuario de HDFS (`HDFS_USER_NAME`)

Asegúrate de que los archivos de entrada estén colocados en la ruta `home/ubuntu/bigdata/examples/` según la estructura indicada arriba.

### 3. Construir y arrancar todo automáticamente

```bash
make start
```

Este comando:

- Construye la imagen Docker (si no existe)
- Lanza el contenedor en segundo plano
- Formatea el NameNode (la primera vez)
- Arranca los servicios de NameNode y DataNode
- Ejecuta automáticamente el script `actividad7.sh`, que:
  - Crea directorios en HDFS (`/books`, `/user/...`)
  - Carga los libros en `/books`
  - Descomprime y sube el archivo `fichero_result.txt`
  - Mueve `frankenstein.txt` dentro de HDFS y lo descarga al host

### 4. Verificar ejecución del script

Para revisar la salida del script de carga automática:

```bash
make actividad-log
```

---

## Comandos útiles

- `make bash` → Accede al contenedor con bash
- `make stop` → Detiene el contenedor
- `make restart` → Reinicia el contenedor
- `make logs` → Muestra los logs
- `make stats` → Monitoriza recursos del contenedor
- `make clean` → Elimina la imagen Docker
- `make prune` → Limpia redes huérfanas de Docker
- `make reset` → Borra contenedor, imagen y archivos generados

---

## Acceso a la interfaz web de HDFS

Abre [http://localhost:50070](http://localhost:50070) en tu navegador para acceder a la interfaz del NameNode.

---

## Notas

- Basado en una instalación manual de Hadoop 2.7.4 sobre Ubuntu 20.04.
- Si estás usando un Mac con chip M1, M2 o M3 (ARM64), asegúrate de definir `PLATFORM=linux/amd64` en tu archivo `.env`. Docker usará emulación automáticamente.
