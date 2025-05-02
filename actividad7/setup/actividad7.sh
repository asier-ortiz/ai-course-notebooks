#!/bin/bash

# Cargar variables del entorno desde .env si no están ya en el entorno
if [ -z "$HDFS_USER_NAME" ] && [ -f "/.env" ]; then
    while IFS='=' read -r key value; do
        if [[ -n "$key" && "$key" != \#* ]]; then
            export "$key=$value"
        fi
    done < /.env
fi

# Asignar variables con valores por defecto si no están definidas
USER_NAME="${HDFS_USER_NAME:-default_user}"
INPUT_DIR="${INPUT_DIR:-/home/ubuntu/bigdata/examples}"
USER_DIR="/user/ubuntu/$USER_NAME"

# Funciones auxiliares
exists_hdfs_dir() { hdfs dfs -test -d "$1"; return $?; }
exists_hdfs_file() { hdfs dfs -test -e "$1"; return $?; }

# Crear directorios si no existen
if ! exists_hdfs_dir "/books"; then
    echo "Creando /books..."
    hdfs dfs -mkdir /books
fi

if ! exists_hdfs_dir "$USER_DIR"; then
    echo "Creando $USER_DIR..."
    hdfs dfs -mkdir -p "$USER_DIR"
fi

# Subir libros
for FILE in frankenstein.txt Quijote.txt; do
    if ! exists_hdfs_file "/books/$FILE"; then
        echo "Subiendo $FILE a /books..."
        if cp "$INPUT_DIR/books/$FILE" /tmp; then
            hdfs dfs -put /tmp/$FILE /books/
        else
            echo "No se pudo copiar $FILE desde $INPUT_DIR/books/"
        fi
    else
        echo "$FILE ya existe en /books"
    fi
done

# Descomprimir y subir fichero_result.txt
if ! exists_hdfs_file "$USER_DIR/fichero_result.txt"; then
    echo "Descomprimiendo fichero_result.zip y buscando fichero_result.txt..."

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
    echo "fichero_result.txt ya existe en $USER_DIR"
fi

# Mover Frankenstein si corresponde
if exists_hdfs_file "/books/frankenstein.txt" && ! exists_hdfs_file "$USER_DIR/frankenstein.txt"; then
    echo "Moviendo frankenstein.txt a $USER_DIR..."
    hdfs dfs -mv /books/frankenstein.txt "$USER_DIR"
fi

# Descargar Frankenstein.txt desde HDFS al host
if [ ! -f "$INPUT_DIR/hdfs/frankenstein.txt" ]; then
    echo "Descargando frankenstein.txt al host..."
    hdfs dfs -get "$USER_DIR/frankenstein.txt" "$INPUT_DIR/hdfs/"
fi

# Resumen
echo "---------------------------"
echo "Resumen de carga en HDFS"
hdfs dfs -ls /books
hdfs dfs -ls "$USER_DIR"