#!/usr/bin/env python3
"""
Script para crear dataset de batch prediction a partir del dataset limpio
Autor: Asier Ortiz
Fecha: 2025-06-14
"""

import pandas as pd
import os


def create_batch_prediction_csv():
    """
    Crea un CSV para batch prediction eliminando la columna 'price'
    del dataset limpio y guardándolo en la carpeta data/
    """

    # Rutas de archivos
    input_file = os.path.join('data', 'madrid_rent_cleaned.csv')
    output_file_full = os.path.join('data', 'madrid_rent_batch_prediction.csv')
    output_file_sample = os.path.join('data', 'madrid_rent_batch_sample.csv')

    try:
        # Verificar que existe el archivo de entrada
        if not os.path.exists(input_file):
            print(f"Error: No se encuentra el archivo {input_file}")
            print("Asegúrate de estar en el directorio 'entrega4' y que el archivo existe")
            return

        print("Cargando dataset limpio...")
        df = pd.read_csv(input_file)

        print(f"Dataset original: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"Columnas: {list(df.columns)}")

        # Verificar que existe la columna 'price'
        if 'price' not in df.columns:
            print("Error: No se encuentra la columna 'price' en el dataset")
            return

        # Crear dataset para batch prediction (sin la columna 'price')
        batch_df = df.drop(columns=['price'])

        print(f"\nDataset para batch prediction: {batch_df.shape[0]} filas, {batch_df.shape[1]} columnas")
        print(f"Columnas para batch: {list(batch_df.columns)}")

        # Guardar dataset completo para batch
        batch_df.to_csv(output_file_full, index=False)
        print(f"Guardado dataset completo: {output_file_full}")

        # Crear también una muestra pequeña para pruebas rápidas (500 filas)
        sample_size = min(500, len(batch_df))
        batch_sample = batch_df.head(sample_size)
        batch_sample.to_csv(output_file_sample, index=False)
        print(f"Guardado dataset muestra ({sample_size} filas): {output_file_sample}")

        # Mostrar estadísticas
        print(f"\nResumen de archivos creados:")
        print(f"   - {output_file_full}: {batch_df.shape[0]} propiedades")
        print(f"   - {output_file_sample}: {batch_sample.shape[0]} propiedades")

        # Mostrar preview de los datos
        print(f"\nPreview del dataset de batch prediction:")
        print(batch_df.head(3).to_string())

        print(f"\nArchivos creados")
        print(f"para realizar batch predictions:")
        print(f"   - Para pruebas rápidas: {output_file_sample}")
        print(f"   - Para análisis completo: {output_file_full}")

    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado - {e}")
    except pd.errors.EmptyDataError:
        print(f"Error: El archivo {input_file} está vacío")
    except Exception as e:
        print(f"Error inesperado: {e}")


if __name__ == "__main__":
    print("Creando dataset para batch prediction...")
    print("=" * 50)
    create_batch_prediction_csv()
    print("=" * 50)
