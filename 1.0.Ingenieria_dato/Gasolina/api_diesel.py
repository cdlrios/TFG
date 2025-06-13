import requests
import csv
from datetime import datetime, timedelta

def obtener_id_municipio(nombre_municipio):
    url_municipios = 'https://sedeaplicaciones.minetur.gob.es/ServiciosRESTCarburantes/PreciosCarburantes/Listados/Municipios/'
    response = requests.get(url_municipios)
    if response.status_code == 200:
        municipios = response.json()
        for municipio in municipios:
            if municipio['Municipio'].lower() == nombre_municipio.lower():
                return municipio['IDMunicipio']
    return None

def obtener_id_producto(nombre_producto):
    url_productos = 'https://sedeaplicaciones.minetur.gob.es/ServiciosRESTCarburantes/PreciosCarburantes/Listados/ProductosPetroliferos/'
    response = requests.get(url_productos)
    if response.status_code == 200:
        productos = response.json()
        for producto in productos:
            if producto['NombreProducto'].lower() == nombre_producto.lower():
                return producto['IDProducto']
    return None

def obtener_precio_promedio(fecha, id_municipio, id_producto):
    url = f'https://sedeaplicaciones.minetur.gob.es/ServiciosRESTCarburantes/PreciosCarburantes/EstacionesTerrestresHist/FiltroMunicipioProducto/{fecha}/{id_municipio}/{id_producto}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        estaciones = data.get('ListaEESSPrecio', [])
        
        precios = [float(estacion['PrecioProducto'].replace(',', '.')) for estacion in estaciones if estacion.get('PrecioProducto')]
        if precios:
            return sum(precios) / len(precios)
    return None

def main():
    nombre_municipio = 'Madrid'
    nombre_producto = 'Gasóleo A habitual'
    fecha_inicio = '01-01-2015'  # Formato dd-mm-aaaa
    fecha_fin = '31-12-2024'
    
    id_municipio = obtener_id_municipio(nombre_municipio)
    id_producto = obtener_id_producto(nombre_producto)
    
    if not id_municipio or not id_producto:
        return
    
    fecha_actual = datetime.strptime(fecha_inicio, '%d-%m-%Y')
    fecha_fin = datetime.strptime(fecha_fin, '%d-%m-%Y')
    
    resultados = []
    
    while fecha_actual <= fecha_fin:
        fecha_str = fecha_actual.strftime('%d-%m-%Y')
        precio_promedio = obtener_precio_promedio(fecha_str, id_municipio, id_producto)
        
        if precio_promedio is not None:
            resultados.append([fecha_str, round(precio_promedio, 3)])
        else:
            resultados.append([fecha_str, 'No disponible'])
        
        fecha_actual += timedelta(days=1)

    # Exportar a CSV
    with open('precios_diesel_madrid.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Fecha', 'Precio Promedio (€/L)'])
        csv_writer.writerows(resultados)

if __name__ == '__main__':
    main()
