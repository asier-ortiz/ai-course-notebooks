# Actividad 7 – HDFS

## 1. Introducción y entorno de trabajo

Esta actividad consiste en realizar una serie de operaciones básicas sobre el sistema de ficheros **HDFS**, como la creación de directorios, la carga de ficheros y su posterior procesamiento o movimiento dentro del sistema.

Aunque el ejercicio propone trabajar sobre un entorno preconfigurado con múltiples servicios Hadoop (NameNode, DataNode, ResourceManager, NodeManager, HistoryServer, etc.), para esta entrega se ha optado por un entorno **Docker personalizado y mínimo**, que incluye únicamente los servicios esenciales: **NameNode** y **DataNode**.

**Justificación de esta decisión:**

- La actividad se centra exclusivamente en operaciones sobre HDFS.
- El resto de servicios (YARN, HistoryServer…) no son necesarios.
- El entorno oficial da errores de arranque en sistemas como macOS con chip M1.
- Se buscaba una solución limpia, reproducible y multiplataforma (ARM y AMD).
- Además, el desarrollo manual ha servido para repasar Docker y reforzar conocimientos.

---

## 2. Pasos realizados manualmente

### Paso 1 – Arranque del servicio HDFS

Se ha usado el comando:

```bash
make start
```

Este construye la imagen (si no existe) y lanza el contenedor en segundo plano.

El script `entrypoint.sh`:

- Formatea el NameNode si es necesario.
- Arranca los procesos de NameNode y DataNode.

Para comprobar que ambos servicios están activos:

```bash
make bash
jps
```

---

### Paso 2 – Crear el directorio `/books`

```bash
hdfs dfs -mkdir /books
```

---

### Paso 3 – Crear directorio personal

```bash
hdfs dfs -mkdir /user/ubuntu/asier_ortiz
```

---

### Paso 4 – Subir libros al HDFS

Se copian `frankenstein.txt` y `Quijote.txt` desde:

```
/home/ubuntu/bigdata/examples/books
```

…al directorio `/books` en HDFS:

```bash
hdfs dfs -put frankenstein.txt /books
hdfs dfs -put Quijote.txt /books
```

---

### Paso 5 – Subir fichero resultante

- Se descomprime `fichero_result.zip` desde:

```
/home/ubuntu/bigdata/examples/hdfs/
```

- Se sube el archivo `fichero_result.txt` extraído al directorio personal en HDFS:

```bash
hdfs dfs -put fichero_result.txt /user/ubuntu/asier_ortiz
```

---

### Paso 6 – Mover `frankenstein.txt` al directorio personal

```bash
hdfs dfs -mv /books/frankenstein.txt /user/ubuntu/asier_ortiz
```

---

### Paso 7 – Descargar archivo desde HDFS

```bash
hdfs dfs -get /user/ubuntu/asier_ortiz/frankenstein.txt /home/ubuntu/bigdata/examples/hdfs/
```

---

### Paso 8 – Verificación desde la interfaz web

Se ha verificado manualmente el estado de los archivos en la interfaz web de Hadoop.

---

## 3. Automatización con Docker y Makefile

Una vez validados todos los pasos manuales, se ha creado un entorno automatizado con Docker que reproduce la misma lógica:

### Incluye:

- `Dockerfile` basado en **Ubuntu 20.04 + Hadoop 2.7.4**
- Configuraciones personalizadas en:
  - `core-site.xml`
  - `hdfs-site.xml`
- Script `entrypoint.sh` que arranca los servicios y ejecuta `actividad7.sh`
- Script `actividad7.sh` con todas las tareas del ejercicio
- `Makefile` con comandos para:
  - construir
  - lanzar
  - detener
  - reiniciar el entorno
- `docker-compose.yml` con la definición del servicio `hdfs-actividad7`, incluyendo volumen, puertos y variables de entorno

---

### Repositorio completo:

🔗 [GitHub – actividad7](https://github.com/asier-ortiz/ai-course-notebooks/tree/main/actividad7)
