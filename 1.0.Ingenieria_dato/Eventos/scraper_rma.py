import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Base URL de la página web de FBref con los partidos del Getafe
base_url = "https://fbref.com/es/equipos/7848bd64/{}/partidos/all_comps/schedule/Resultados-y-partidos-de-Getafe-Todas-las-competencias"
# Rango de temporadas
temp_start = 2014
temp_end = 2025

data = []

# Iterar sobre cada temporada
temp_range = [f"{year}-{year+1}" for year in range(temp_start, temp_end)]
for season in temp_range:
    url = base_url.format(season)
    print(f"Extrayendo datos de la temporada {season}...")
    
    # Hacer la solicitud HTTP para obtener el contenido de la web
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # Encontrar la tabla de partidos
        table = soup.find("table", {"id": "matchlogs_for"})

        # Extraer los encabezados de la tabla para localizar la columna "Asistencia"
        if table:
            headers_table = [th.text.strip() for th in table.find_all("th")]
            attendance_index = headers_table.index("Asistencia") if "Asistencia" in headers_table else None

            rows = table.find_all("tr")[1:]  # Omitir la cabecera
            for row in rows:
                columns = [td.text.strip() for td in row.find_all("td")]
                if len(columns) > 8 and attendance_index is not None:
                    # Extraer la fecha desde el <th> con el atributo data-stat="date"
                    date_cell = row.find("th", {"data-stat": "date"})
                    date = date_cell.text.strip() if date_cell else "N/A"
                    
                    venue = columns[4]      # Sede (Local, Visitante o Neutral)
                    opponent = columns[8]   # Oponente
                    attendance = columns[attendance_index - 1] if attendance_index - 1 < len(columns) else "N/A"
                    if venue == "Local":
                        data.append((season, date, opponent, attendance))
    else:
        print(f"Error al acceder a la URL de la temporada {season}. Código de estado: {response.status_code}")
    
    # Temporizador para evitar bloqueos
    time.sleep(5)  # Esperar 5 segundos entre cada solicitud

# Crear DataFrame con la información correcta
df_local = pd.DataFrame(data, columns=["Temporada", "Fecha", "Rival", "Espectadores"])

# Guardar el resultado en un archivo CSV
df_local.to_csv("partidos_Getafe.csv", index=False)

