import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL de la página que quieres scrape.
url = "https://x-y.es/aemet/est-3195-madrid-retiro?fecha=2024-10-12"

# Realizar la solicitud GET a la página.
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')

    # Encuentra la tabla de datos (ajustar según la estructura de la página).
    table = soup.find('table')  # Modifica el selector según la estructura HTML.

    if table:
        # Extraer los encabezados de la tabla.
        headers = [th.text.strip() for th in table.find_all('th')]

        # Extraer los datos de cada fila.
        rows = []
        for row in table.find_all('tr')[1:]:  # Excluir encabezados
            cells = [td.text.strip() for td in row.find_all('td')]
            if cells:  # Si la fila contiene datos
                rows.append(cells)

        # Crear un DataFrame.
        
        df = pd.DataFrame(rows, columns=headers)

        # Guardar como archivo CSV.
        df.to_csv('factores_meteorologicos.csv', index=False)
        print("Datos guardados en 'factores_meteorologicos.csv'")
    else:
        print("No se encontró ninguna tabla en la página.")
else:
    print(f"Error al acceder a la página: {response.status_code}")
