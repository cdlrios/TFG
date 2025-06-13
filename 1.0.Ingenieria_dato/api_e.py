import requests

API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjYXJsb3MuZGVsb3NyaW9zbW91dmV0MkBnbWFpbC5jb20iLCJqdGkiOiIxNmU1ZjA3OS03NzdhLTQ0OTgtOTU4ZC0yY2E1YWM3M2M1MTIiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTczMjEyNTAwNywidXNlcklkIjoiMTZlNWYwNzktNzc3YS00NDk4LTk1OGQtMmNhNWFjNzNjNTEyIiwicm9sZSI6IiJ9.F_dsnr28mYlkS597RvoWV44HG2vYghAcNa83DrL0nWY"
INVENTARIO_URL = "https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones"

response = requests.get(INVENTARIO_URL, params={"api_key": API_KEY})
if response.status_code == 200:
    data = response.json()
    datos_url = data.get("datos")
    if datos_url:
        response_datos = requests.get(datos_url)
        if response_datos.status_code == 200:
            estaciones = response_datos.json()
            # Filtra las estaciones que contienen "Madrid" en su nombre
            estaciones_madrid = [est for est in estaciones if "Madrid" in est['nombre']]
            for est in estaciones_madrid:
                print(f"IDEMA: {est['indicativo']}, Nombre: {est['nombre']}, Latitud: {est['latitud']}, Longitud: {est['longitud']}")
        else:
            print("Error al acceder a los datos del inventario:", response_datos.status_code)
    else:
        print("No se encontró la URL de datos en la respuesta.")
else:
    print("Error al conectar con la API:", response.status_code)
