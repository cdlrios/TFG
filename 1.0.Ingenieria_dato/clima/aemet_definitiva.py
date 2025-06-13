import requests
import csv
import time
import random
from datetime import datetime, timedelta

# Tu clave API de AEMET
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjYXJsb3MuZGVsb3NyaW9zbW91dmV0MkBnbWFpbC5jb20iLCJqdGkiOiIxNmU1ZjA3OS03NzdhLTQ0OTgtOTU4ZC0yY2E1YWM3M2M1MTIiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTczMjEyNTAwNywidXNlcklkIjoiMTZlNWYwNzktNzc3YS00NDk4LTk1OGQtMmNhNWFjNzNjNTEyIiwicm9sZSI6IiJ9.F_dsnr28mYlkS597RvoWV44HG2vYghAcNa83DrL0nWY"

# Endpoint de valores climatológicos diarios
BASE_URL = "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{start_date}/fechafin/{end_date}/estacion/{station_id}/"

# Función para obtener datos en intervalos de 6 meses con espera aleatoria
def obtener_climatologia_periodos(start_date, end_date, estacion_id, nombre_archivo):
    fecha_inicio = datetime.strptime(start_date, "%Y-%m-%d")
    fecha_fin = datetime.strptime(end_date, "%Y-%m-%d")
    intervalo = timedelta(days=180)  # 6 meses

    while fecha_inicio < fecha_fin:
        fecha_final_intervalo = min(fecha_inicio + intervalo, fecha_fin)
        obtener_climatologia_rango(fecha_inicio.strftime("%Y-%m-%d"), fecha_final_intervalo.strftime("%Y-%m-%d"), estacion_id, nombre_archivo)

        fecha_inicio = fecha_final_intervalo + timedelta(days=1)  # Continuar desde el día siguiente
        
        # Espera aleatoria entre 20 y 30 segundos
        tiempo_espera = random.randint(25, 38)
        print(f"Esperando {tiempo_espera} segundos antes de la siguiente consulta...")
        time.sleep(tiempo_espera)

# Función para obtener datos de la API para un rango específico
def obtener_climatologia_rango(fechainicio, fechafin, estacion_id, nombre_archivo):
    try:
        url = BASE_URL.format(start_date=f"{fechainicio}T00:00:00UTC", end_date=f"{fechafin}T23:59:59UTC", station_id=estacion_id)
        print(f"Generando solicitud a la URL: {url}")

        params = {"api_key": API_KEY}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            datos_url = data.get("datos")

            if datos_url:
                print(f"Accediendo a los datos en: {datos_url}")
                response_datos = requests.get(datos_url)
                if response_datos.status_code == 200:
                    datos = response_datos.json()
                    guardar_en_csv(datos, nombre_archivo)
                    return
                else:
                    print(f"Error al acceder a los datos: {response_datos.status_code}")
            else:
                print("Error: No se encontró la URL de datos.")
        else:
            print(f"Error al conectar con la API: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Error de conexión: {e}")

# Función para guardar los datos en CSV
def guardar_en_csv(datos, nombre_archivo):
    if not datos:
        print("No hay datos para guardar.")
        return

    # Si el archivo no existe, creamos una cabecera
    try:
        with open(nombre_archivo, mode='r', encoding='utf-8') as f:
            existe_archivo = True
    except FileNotFoundError:
        existe_archivo = False

    with open(nombre_archivo, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Escribir la cabecera si el archivo no existe
        if not existe_archivo:
            writer.writerow(["Fecha", "Temperatura Máxima (°C)", "Temperatura Media (°C)", "Temperatura Mínima (°C)", 
                             "Precipitaciones (mm)", "Velocidad Media Viento (km/h)", "Humedad Relativa Media (%)"])

        for registro in datos:
            writer.writerow([
                registro.get("fecha", "N/A"),
                registro.get("tmax", "N/A"),
                registro.get("tmed", "N/A"),
                registro.get("tmin", "N/A"),
                registro.get("prec", "N/A"),
                registro.get("velmedia", "N/A"),
                registro.get("hrMedia", "N/A")
            ])

    print(f"Datos guardados en {nombre_archivo}")

# Llamar a la función para obtener datos desde 2015 hasta 2024
obtener_climatologia_periodos("2015-01-01", "2024-12-31", "3125Y", "climatologia_SSReyes.csv")
