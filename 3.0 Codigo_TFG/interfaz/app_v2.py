
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import pickle
from datetime import date
import os

# ===== CONFIGURACIÓN GENERAL =====
st.set_page_config(layout="wide")
st.title("TFG Metro Madrid — Dashboard con Calendario de Festivos")

# ===== RUTAS =====
DATA_DIR = "interfaz"
ESTACIONES_FILE = os.path.join(DATA_DIR, "estaciones_dem.csv")
CALENDARIO_FILE = os.path.join(DATA_DIR, "calendario_streamlit.csv")
MODELOS_DIR = "modelos_xgb"

# ===== CARGA DE DATOS =====
df = pd.read_csv(ESTACIONES_FILE, sep=";", decimal=",")

# ===== CARGA Y PREPARACIÓN DEL CALENDARIO =====
cal = pd.read_csv(
    CALENDARIO_FILE,
    sep=";",
    decimal=",",
    parse_dates=["Fecha"],
    dayfirst=True
)
# Renombrar columnas festivos
festivos_map = {
    'NU_DIA_MES':'nu_dia_mes','NU_MES':'nu_mes','NU_DIA_SEM':'nu_dia_sem',
    'Navidad':'in_navidad','Año_nuevo':'in_anonuevo','Reyes':'in_reyesmagos',
    'Primero_Mayo':'in_primeromayo','Fiesta_CAM':'in_2demayo','San_Isidro':'in_sanisidro',
    'Asuncion':'in_asuncion','Hispanidad':'in_12oct','Todos_santos':'in_1nov',
    'Almudena':'in_almudena','Constitucion':'in_constitucion','Inmaculada_Concepcion':'in_purisima',
    'Jueves_Santo':'in_juesanto','Viernes_Santo':'in_viesanto','Santiago_Apostol':'in_santiago',
    'BLACK_FRIDAY':'in_blackfriday','NU_SEMANA':'nu_semana'
}
cal.rename(columns=festivos_map, inplace=True)
# Rango completo fechas
dates = pd.date_range(start="2025-01-01", end="2026-12-31", freq='D')
cal.set_index('Fecha', inplace=True)
cal = cal.reindex(dates, fill_value=0)
cal.index.name = 'Fecha'

# ===== CARGA DINÁMICA DE MODELOS =====
@st.cache_resource
def load_model(cod_est):
    path = os.path.join(MODELOS_DIR, f"modelo_{cod_est}.pkl")
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Modelo no encontrado: {path}")
        return None

# ===== NOMBRE DE FEATURES =====
FEATURES = [
    'nu_dia_mes','nu_mes','nu_dia_sem','in_laborable','in_lectivo',
    'in_navidad','in_anonuevo','in_reyesmagos','in_primeromayo','in_2demayo',
    'in_sanisidro','in_asuncion','in_12oct','in_1nov','in_almudena','in_constitucion',
    'in_purisima','in_juesanto','in_viesanto','in_santiago','in_blackfriday',
    'nu_semana','precio_gasolina','precio_diesel','t_med','precipitaciones',
    'vel_med_viento','hum_rel_med','ev_bernabeu','ev_ifema','ev_ventas',
    'ev_metropolitano','ev_movistararena','ev_vallecas','ev_vistalegre','cof_cierre'
]

# ===== SIDEBAR =====
page = st.sidebar.selectbox("Ir a", ["Mapa","Formulario"])

# --------------------------------------------------
# PÁGINA MAPA
# --------------------------------------------------
if page == "Mapa":
    st.header("🗺️ Mapa de estaciones")
    mapa = folium.Map(location=[40.4168,-3.7038], zoom_start=11)
    cluster = MarkerCluster().add_to(mapa)
    for _,r in df.iterrows():
        html = f"<b>{r['nom_est']}</b><br><a href='?estacion={r['cod_est']}'>Predicción</a>"
        folium.Marker([r['Latitud'],r['Longitud']], popup=html).add_to(cluster)
    st_folium(mapa)

# --------------------------------------------------
# PÁGINA FORMULARIO
# --------------------------------------------------
else:
    st.header("📝 Formulario de predicción")
    # Estación y modelo
    est = st.selectbox("Estación", options=df['nom_est'])
    cod = df[df['nom_est']==est]['cod_est'].iloc[0]
    model = load_model(cod)
    # Fecha y festivos
    fecha = st.date_input("Fecha", min_value=date(2025,1,1), max_value=date(2026,12,31))
    ts = pd.to_datetime(fecha)
    row = cal.loc[ts]
    # Climatología y precios
    t = st.slider("Temp (°C)",-5,45,20)
    pcp = st.slider("Precipitaciones (mm)",0,100,0)
    v = st.slider("Viento (km/h)",0,100,10)
    hr = st.slider("Humedad (%)",0,100,50)
    pg = st.number_input("Gasolina (€)",0.0,5.0,1.5,0.01)
    pdl = st.number_input("Diésel (€)",0.0,5.0,1.4,0.01)
    # Eventos
    ev = {stadio: st.checkbox(f"Ev {stadio}") for stadio in ['Bernabeu','IFEMA','Ventas','Metropolitano','MovistarArena','Vallecas','VistaAlegre']}
    asis = {f"asistentes_{k.lower()}": st.number_input(f"Asistentes {k}",0,step=500) if ev[k] else 0 for k in ev}
    # Cof cierre hardcodeado a 0
    cof = 0
    if st.button("Predecir") and model:
        # Armar X_pred
        data = {}
        for f in FEATURES:
            if f in row.index:
                data[f] = int(row[f])
            elif f=='in_laborable':
                data[f] = int(fecha.weekday()<5)
            elif f=='in_lectivo':
                data[f] = data['in_laborable']
            elif f=='precio_gasolina': data[f]=pg
            elif f=='precio_diesel': data[f]=pdl
            elif f=='t_med': data[f]=t
            elif f=='precipitaciones': data[f]=pcp
            elif f=='vel_med_viento': data[f]=v
            elif f=='hum_rel_med': data[f]=hr
            elif f.startswith('ev_'):
                key = f.split('ev_')[1]
                keymap = {
                    'bernabeu':'Bernabeu','ifema':'IFEMA','ventas':'Ventas',
                    'metropolitano':'Metropolitano','movistararena':'MovistarArena',
                    'vallecas':'Vallecas','vistalegre':'VistaAlegre'
                }
                data[f] = int(ev[keymap[key]])
            elif f=='cof_cierre': data[f]=cof
            else:
                # asistentes
                data[f] = asis.get(f,0)
        X_pred = pd.DataFrame([data], columns=FEATURES)
        try:
            pred = model.predict(X_pred)[0]
            st.success(f"Predicción: {int(pred):,} entradas")
        except Exception as e:
            st.error(f"Error: {e}")
