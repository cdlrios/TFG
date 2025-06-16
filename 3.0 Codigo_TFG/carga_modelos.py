import os
import joblib
import pandas as pd
from xgboost import XGBRegressor, Booster

# Ruta al .pkl de tu modelo
MODEL_PATH = 'modelos_xgb/modelo_708.pkl'

# 1. Cargar el modelo
model = joblib.load(MODEL_PATH)

# 2. Mostrar información general
print("Tipo de objeto cargado:", type(model))
print("\n--- Parámetros del modelo XGBRegressor ---")
for k, v in model.get_params().items():
    print(f"{k}: {v}")

# 3. Información de las características (si existe)
if hasattr(model, 'feature_names_in_'):
    print("\n--- Nombres de features ---")
    print(model.feature_names_in_)
elif hasattr(model, 'feature_names_'):
    print("\n--- Nombres de features ---")
    print(model.feature_names_)
else:
    print("\nNo hay atributo de nombres de features en el objeto.")

# 4. Importancia de las features
if hasattr(model, 'feature_importances_'):
    print("\n--- Feature importances (normalizadas) ---")
    for name, imp in zip(
        getattr(model, 'feature_names_in_', getattr(model, 'feature_names_', range(len(model.feature_importances_)))),
        model.feature_importances_
    ):
        print(f"{name}: {imp:.4f}")
else:
    print("\nEste modelo no expone `feature_importances_`.")

# 5. Obtener el Booster interno y revisar sus detalles
booster: Booster = model.get_booster()
print("\n--- Detalles del Booster interno ---")
print("Número de árboles:", len(booster.get_dump()))
print("Atributos adicionales guardados en el booster:")
print(booster.attributes())

# 6. (Opcional) Volcar el primer árbol como texto
print("\n--- Primer árbol (primeros 20 nodos) ---")
dump0 = booster.get_dump(with_stats=False)[0].splitlines()
for line in dump0[:20]:
    print(line)
