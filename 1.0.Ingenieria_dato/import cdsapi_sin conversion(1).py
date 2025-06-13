import cdsapi
import os
import time

def download_era5_data(year, month, day, variables, output_filename):
    c = cdsapi.Client()
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': variables,
                'year': str(year),
                'month': f"{month:02d}",
                'day': f"{day:02d}",
                'time': [f"{hour:02d}:00" for hour in range(24)],  
                'area': [35.89, -1.89, 42.36, 5],  # Región acotada
            },
            output_filename
        )
    except Exception as e:
        print(f"Error en la descarga de {output_filename}: {e}")
        print("Esperando 60 segundos antes de reintentar...")
        time.sleep(60)  
        download_era5_data(year, month, day, variables, output_filename)

def get_days_in_month(year, month):
    """Devuelve el número de días en un mes dado un año."""
    if month == 2:
        return 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
    elif month in [4, 6, 9, 11]:
        return 30
    else:
        return 31

output_directory = r"C:\Users\pablo\OneDrive - UFV\Cuarto Año\TFG\dana\Datos_era5_2"

os.makedirs(output_directory, exist_ok=True)

# Variables a descargar
variables = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_dewpoint_temperature',
    '2m_temperature',
    'mean_sea_level_pressure',
    'sea_surface_temperature',
    'surface_pressure',
    'total_precipitation',
    'significant_height_of_combined_wind_waves_and_swell',
    'convective_available_potential_energy',
    'mean_wave_direction'
]


start_year = 2014
start_month = 10
start_day = 31

for year in range(start_year, 2024):
    for month in range(start_month if year == start_year else 1, 13):
        # Obtener el número de días en el mes actual
        days_in_month = get_days_in_month(year, month)
        
        for day in range(start_day if year == start_year and month == start_month else 1, days_in_month + 1):
            filename = os.path.join(output_directory, f'era5_data_{year}_{month:02d}_{day:02d}.nc')
            
            # Descarga de datos
            download_era5_data(year, month, day, variables, filename)
            
            # Añadir una pausa de 30 segundos entre solicitudes para evitar bloqueos
            print(f"Descarga completada para {filename}. Esperando 15 segundos antes de la próxima solicitud...")
            time.sleep(15)
            
            # Resetear start_day a 1 después del primer mes completo
            if year == start_year and month == start_month:
                start_day = 1
