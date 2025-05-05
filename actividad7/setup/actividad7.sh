#!/bin/bash

##################################################
# Script para cargar datos del ejercicio en HDFS #
##################################################

# Establezco variables por defecto si no vienen del entorno
USER_NAME="${HDFS_USER_NAME:-default_user}"
INPUT_DIR="${INPUT_DIR:-/home/ubuntu/bigdata/examples}"
USER_DIR="/user/ubuntu/$USER_NAME"

# Funciones auxiliares

# Comprueba si existe un directorio en HDFS
exists_hdfs_dir() {
  hdfs dfs -test -d "$1"
  return $?
}

# Comprueba si existe un archivo en HDFS
exists_hdfs_file() {
  hdfs dfs -test -e "$1"
  return $?
}

# Crear rutas necesarias en HDFS
if ! exists_hdfs_dir "/books"; then
  echo "Creando /books..."
  hdfs dfs -mkdir /books
fi

if ! exists_hdfs_dir "$USER_DIR"; then
  echo "Creando $USER_DIR..."
  hdfs dfs -mkdir -p "$USER_DIR"
fi

# Subir archivos de los libros al HDFS
for FILE in frankenstein.txt Quijote.txt; do
  if ! exists_hdfs_file "/books/$FILE"; then
    echo "Subiendo $FILE a /books..."
    if cp "$INPUT_DIR/books/$FILE" /tmp; then
      hdfs dfs -put /tmp/$FILE /books/
    else
      echo "No se pudo copiar $FILE desde $INPUT_DIR/books/"
    fi
  else
    echo "$FILE ya está en /books"
  fi
done


# Descomprimir fichero_result y subirlo si no existe
if ! exists_hdfs_file "$USER_DIR/fichero_result.txt"; then
  echo "Descomprimiendo fichero_result.zip para subir fichero_result.txt..."

  ZIP_PATH="$INPUT_DIR/hdfs/fichero_result.zip"
  if [ ! -f "$ZIP_PATH" ]; then
    echo "No se encontró $ZIP_PATH"
  else
    unzip -o "$ZIP_PATH" -d /tmp >/dev/null
    FOUND_FILE=$(find /tmp/fichero_result -type f -name "fichero_result.txt" | head -n 1)

    if [ -f "$FOUND_FILE" ]; then
      hdfs dfs -put "$FOUND_FILE" "$USER_DIR/fichero_result.txt"
      rm -f "$FOUND_FILE"
    else
      echo "No se encontró fichero_result.txt dentro del ZIP"
    fi
  fi
else
  echo "fichero_result.txt ya está en $USER_DIR"
fi


# Mover frankenstein.txt al directorio del usuario si es necesario
if exists_hdfs_file "/books/frankenstein.txt" && ! exists_hdfs_file "$USER_DIR/frankenstein.txt"; then
  echo "Moviendo frankenstein.txt a $USER_DIR..."
  hdfs dfs -mv /books/frankenstein.txt "$USER_DIR"
fi


# Descargar frankenstein.txt desde HDFS al host (si no existe)
if [ ! -f "$INPUT_DIR/hdfs/frankenstein.txt" ]; then
  echo "Descargando frankenstein.txt desde HDFS al host..."
  hdfs dfs -get "$USER_DIR/frankenstein.txt" "$INPUT_DIR/hdfs/"
fi


# Mostrar resumen de ficheros en HDFS
echo "---------------------------"
echo "Resumen de carga en HDFS"
hdfs dfs -ls /books
hdfs dfs -ls "$USER_DIR"