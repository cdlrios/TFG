import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import pickle

# ===== CONFIGURACIÓN GENERAL =====
st.set_page_config(layout="wide")
st.title("Predicción de entradas por estación de Metro de Madrid")

# ===== CARGA DE DATOS =====
df = pd.read_csv("estaciones_dem.csv", sep=";", decimal=",")

# ===== CARGA DEL MODELO REAL =====
with open("modelo.pkl", "rb") as f:
    modelo = pickle.load(f)

# ===== CREAR MAPA =====
m = folium.Map(location=[40.4168, -3.7038], zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)

for _, row in df.iterrows():
    popup_html = f"""
    <b>{row['nom_est']}</b><br>
    Población total: {row['pob_tot']}<br>
    Edad media: {row['edad_media']}<br><br>
    <a href='?estacion={row['cod_est']}'>Hacer predicción</a>
    """
    folium.Marker(
        location=[row['Latitud'], row['Longitud']],
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(marker_cluster)

# ===== MOSTRAR MAPA =====
st_folium(m, width=800, height=600)


# ===== GESTIÓN DE ESTACIÓN SELECCIONADA =====
query_params = st.experimental_get_query_params()
cod_est = query_params.get("estacion", [None])[0]

if cod_est:
    st.session_state["cod_est"] = int(cod_est)

cod_est = st.session_state.get("cod_est")

# ===== FORMULARIO DE PREDICCIÓN =====
if cod_est:
    cod_est = int(cod_est)
    est_data = df[df["cod_est"] == cod_est].iloc[0]
    st.subheader(f"Predicción para la estación: {est_data['nom_est']}")

    temperatura = st.slider("Temperatura ambiente (°C)", 0, 45, 20)
    festivo = st.selectbox("¿Es festivo?", [0, 1], format_func=lambda x: "Sí" if x else "No")
    lluvia = st.slider("Lluvia (mm)", 0, 50, 5)

    if st.button("Calcular predicción"):
        X = pd.DataFrame([[temperatura, festivo, lluvia]], columns=["temperatura", "festivo", "lluvia"])
        pred = modelo.predict(X)[0]
        st.success(f"Predicción estimada de entradas: {int(pred):,} personas")

