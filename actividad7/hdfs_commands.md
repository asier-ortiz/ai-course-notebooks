
# HDFS Commands Cheatsheet 📄

Esta es una recopilación de comandos básicos de HDFS junto a una breve explicación de cada uno.

---

## 📁 Gestión de directorios

- **Crear un directorio:**
  ```bash
  hdfs dfs -mkdir /ruta/del/directorio
  ```
  > Crea un nuevo directorio en HDFS.

- **Crear directorios anidados:**
  ```bash
  hdfs dfs -mkdir -p /ruta/padre/hijo
  ```
  > Crea varios directorios a la vez (si no existen los padres).

- **Listar contenidos:**
  ```bash
  hdfs dfs -ls /ruta
  ```
  > Lista los archivos y carpetas en el directorio indicado.

- **Ver información detallada:**
  ```bash
  hdfs dfs -ls -h /ruta
  ```
  > Muestra los tamaños de archivo de forma legible (ej: MB, GB).

- **Eliminar un directorio vacío:**
  ```bash
  hdfs dfs -rmdir /ruta
  ```
  > Elimina un directorio solo si está vacío.

- **Eliminar un directorio con contenido:**
  ```bash
  hdfs dfs -rm -r /ruta
  ```
  > Elimina el directorio y todos los archivos que contenga.

---

## 📄 Gestión de archivos

- **Subir archivo de local a HDFS:**
  ```bash
  hdfs dfs -put archivo_local /ruta/en/hdfs
  ```
  > Copia un archivo local a un directorio en HDFS.

- **Descargar archivo de HDFS a local:**
  ```bash
  hdfs dfs -get /ruta/en/hdfs/archivo /ruta/local
  ```
  > Copia un archivo de HDFS a tu sistema de ficheros local.

- **Mover archivo dentro de HDFS:**
  ```bash
  hdfs dfs -mv /ruta/origen/archivo /ruta/destino
  ```
  > Mueve (o renombra) un archivo o carpeta dentro de HDFS.

- **Eliminar archivo:**
  ```bash
  hdfs dfs -rm /ruta/archivo
  ```
  > Borra un archivo en HDFS.

- **Ver el contenido de un archivo:**
  ```bash
  hdfs dfs -cat /ruta/archivo
  ```
  > Muestra el contenido de un archivo de HDFS en la terminal.

- **Contar archivos, directorios y bytes:**
  ```bash
  hdfs dfs -count /ruta
  ```
  > Muestra número de archivos, directorios y tamaño total de datos.

---

## 🔍 Información adicional

- **Espacio usado en HDFS:**
  ```bash
  hdfs dfsadmin -report
  ```
  > Muestra estadísticas del sistema HDFS (nodos, espacio usado, libre...).

- **Espacio usado por un directorio:**
  ```bash
  hdfs dfs -du -h /ruta
  ```
  > Muestra el tamaño ocupado por cada archivo/directorio dentro de la ruta especificada.

---

## 🛠️ Tip

- Puedes usar `hdfs dfs -help` para ver más opciones en cualquier comando.
