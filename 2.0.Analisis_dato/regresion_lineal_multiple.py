

import pandas as pd

# Cargar datos
df = pd.read_csv("datos_modelo.csv",delimiter=";")

# Ver las primeras filas
print(df.head())


import seaborn as sns
import matplotlib.pyplot as plt

# Revisar estadísticas generales
print(df.describe())

# Mapa de calor para ver correlaciones
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
