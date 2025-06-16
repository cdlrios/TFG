import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import TimeSeriesSplit, HalvingRandomSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===============================
# Script RF Halving para una estación
# ===============================

# Parámetros configurables
file_path = Path("data/dataset_metro.csv")  # Ruta al CSV
station_id = 708                          # Código de estación
corr_threshold = 0.05                      # Umbral de correlación
cv_splits = 5                              # Número de folds para TimeSeriesSplit
random_state = 42                          # Semilla fija
param_dist = {
    'n_estimators': randint(100, 601),
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None]
}

# 1. Carga y filtrado de datos
df = pd.read_csv(file_path, sep=";", decimal=",")
df_sta = df[df["cod_est"] == station_id].drop(columns=["cod_est"]).reset_index(drop=True)
y = df_sta["ca_entradas"].values
X = df_sta.drop(columns=["ca_entradas"])

# 2. Selección por correlación
corr = pd.Series(X.corrwith(df_sta["ca_entradas"]).abs(), index=X.columns)
keep = corr[corr >= corr_threshold].sort_values(ascending=False)
X_sel = X[keep.index]

# Gráfico 1: Correlaciones
plt.figure()
plt.barh(keep.index, keep.values)
plt.gca().invert_yaxis()
plt.title(f"Correlación |r| ≥ {corr_threshold} (Estación {station_id})")
plt.xlabel("Valor absoluto de r")
plt.tight_layout()
plt.show()

# 3. Definición del CV de series temporales
tscv = TimeSeriesSplit(n_splits=cv_splits)

# 4. Configurar HalvingRandomSearchCV
rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
search = HalvingRandomSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    factor=3,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    random_state=random_state,
    n_jobs=-1,
    verbose=1,
    return_train_score=False
)

# 5. Ajuste de búsqueda de hiperparámetros
search.fit(X_sel, y)

# 6. Gráfica de evolución de RMSE vs iteración
results = pd.DataFrame(search.cv_results_)
# Agrupar por iteración y extraer el mejor (menos RMSE)
best_per_iter = results.groupby('iter')['mean_test_score'].max()
# RMSE = -score
plt.figure()
plt.plot(best_per_iter.index, -best_per_iter.values, marker='o')
plt.title("Evolución del RMSE por iteración (HalvingSearch)")
plt.xlabel("Iteración de halving")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()

# 7. Evaluación final con CV manual y predicción completa
y_preds = np.zeros_like(y, dtype=float)
for train_idx, test_idx in tscv.split(X_sel):
    best = search.best_estimator_
    best.fit(X_sel.iloc[train_idx], y[train_idx])
    y_preds[test_idx] = best.predict(X_sel.iloc[test_idx])

# Métricas globales
rmse = np.sqrt(mean_squared_error(y, y_preds))
mae = mean_absolute_error(y, y_preds)
r2 = r2_score(y, y_preds)
print(f"\nMétricas en CV completo:\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}")

# Gráfico 3: Predicción vs Real (todos los pliegues)
plt.figure()
plt.scatter(y, y_preds, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='--')
plt.title("Predicción vs Real (CV completo)")
plt.xlabel("Real")
plt.ylabel("Predicción")
plt.tight_layout()
plt.show()

# 8. Importancias de variables
feat_imp = pd.Series(search.best_estimator_.feature_importances_, index=keep.index)
feat_imp = feat_imp.sort_values(ascending=False)

plt.figure()
plt.barh(feat_imp.index, feat_imp.values)
plt.gca().invert_yaxis()
plt.title("Importancia de Features (Random Forest)")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()

# 9. Parámetros óptimos
print(f"\nParámetros óptimos: {search.best_params_}")

