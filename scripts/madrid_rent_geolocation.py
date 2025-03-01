import os
import time
import json
import pandas as pd
import requests
import concurrent.futures
import kagglehub

# Descargar el dataset desde Kaggle
dataset_path = kagglehub.dataset_download("mapecode/madrid-province-rent-data")

# Cargar el archivo CSV descargado
csv_file = os.path.join(dataset_path, [f for f in os.listdir(dataset_path) if f.endswith(".csv")][0])
df = pd.read_csv(csv_file)

# Configuración de la API de Google Maps
API_KEY = "YOUR_API_KEY"
GOOGLE_MAPS_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# Número máximo de hilos para procesamiento en paralelo
MAX_THREADS = 5

# Archivo de caché para evitar llamadas repetidas a la API
CACHE_FILE = "geolocation_cache.json"

# Cargar caché si existe, de lo contrario, crear un diccionario vacío
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        geolocation_cache = json.load(f)
else:
    geolocation_cache = {}


def get_geolocation(row):
    """Obtiene coordenadas y datos faltantes de la API de Google Maps con caché persistente."""

    # Construcción segura de la dirección evitando valores nulos
    address_parts = [
        str(row["location"]) if pd.notna(row["location"]) else "",
        str(row["subdistrict"]) if pd.notna(row["subdistrict"]) else "",
        str(row["district"]) if pd.notna(row["district"]) else "",
        "Madrid, Spain",
        str(int(row["postalcode"])) if pd.notna(row["postalcode"]) else "",
    ]
    address = ", ".join(filter(None, address_parts))  # Eliminar elementos vacíos

    if not address.strip():  # Evita direcciones completamente vacías
        return None, None, None, None, None

    # Si la dirección ya está en la caché, devolver los valores guardados
    if address in geolocation_cache:
        return geolocation_cache[address]

    params = {"address": address, "key": API_KEY}

    retries = 5  # Más intentos para mayor estabilidad
    for attempt in range(retries):
        try:
            response = requests.get(GOOGLE_MAPS_URL, params=params, timeout=10)
            data = response.json()

            if "results" in data and len(data["results"]) > 0:
                location = data["results"][0]["geometry"]["location"]
                lat, lng = location["lat"], location["lng"]

                # Extraer dirección formateada y componentes
                address_components = data["results"][0]["address_components"]
                postal_code, district, subdistrict = None, None, None

                for component in address_components:
                    if "postal_code" in component["types"]:
                        postal_code = component["long_name"]
                    elif "administrative_area_level_2" in component["types"]:  # Distrito
                        district = component["long_name"]
                    elif "sublocality" in component["types"]:  # Subdistrito / Barrio
                        subdistrict = component["long_name"]

                # Guardar en caché para futuras consultas
                geolocation_cache[address] = (lat, lng, postal_code, district, subdistrict)

                return lat, lng, postal_code, district, subdistrict

        except requests.exceptions.ConnectionError:
            time.sleep(2 ** attempt)  # Espera exponencial antes de reintentar
        except requests.exceptions.Timeout:
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)

    return None, None, None, None, None


# Dividir el dataset en partes para procesar en paralelo
df_chunks = [df.iloc[i::MAX_THREADS].copy() for i in range(MAX_THREADS)]


def process_chunk(df_chunk):
    """Procesa un fragmento del dataset en un hilo."""
    results = df_chunk.apply(lambda row: pd.Series(get_geolocation(row)), axis=1)

    # Evitar errores si la API devuelve None
    results.fillna(value={"lat": None, "lng": None, "postalcode_filled": None, "district_filled": None,
                          "subdistrict_filled": None}, inplace=True)

    df_chunk[["lat", "lng", "postalcode_filled", "district_filled", "subdistrict_filled"]] = results
    return df_chunk


# Ejecutar procesamiento en paralelo
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = [executor.submit(process_chunk, chunk) for chunk in df_chunks]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

# Combinar resultados de los hilos
df_final = pd.concat(results)

# Rellenar valores nulos en el dataset original
df_final["postalcode"] = df_final["postalcode"].fillna(df_final["postalcode_filled"])
df_final["district"] = df_final["district"].fillna(df_final["district_filled"])
df_final["subdistrict"] = df_final["subdistrict"].fillna(df_final["subdistrict_filled"])

# Eliminar columnas auxiliares
df_final.drop(columns=["postalcode_filled", "district_filled", "subdistrict_filled"], inplace=True)

# Guardar la caché de resultados para futuras ejecuciones
with open(CACHE_FILE, "w") as f:
    json.dump(geolocation_cache, f)

# Guardar el dataset actualizado en el mismo directorio
script_directory = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_directory, "madrid_rent_with_geolocation.csv")
df_final.to_csv(output_file, index=False)
