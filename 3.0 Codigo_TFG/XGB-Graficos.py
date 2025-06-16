import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import shapiro
from xgboost import XGBRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import TimeSeriesSplit, HalvingGridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sklearn

# ===============================
# Script XGBoost Halving para una estación
# ===============================

# 0. Reproducibilidad y versiones
np.random.seed(42)
print(f"scikit‑learn {sklearn.__version__}, xgboost {XGBRegressor.__module__.split('.')[0]}")

# Parámetros
file_path = Path("data/dataset_metro.csv")  # Ajusta la ruta a tus datos
station_id = 708                          # Código de estación a analizar
corr_threshold = 0.05                      # Umbral de correlación
cv_splits = 5                              # Folds para TimeSeriesSplit
random_state = 42                          # Semilla
param_grid = {
    'n_estimators': [100, 300, 600],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 8]
}

# 1. Carga y filtrado de datos para la estación
df = pd.read_csv(file_path, sep=';', decimal=',')
df_sta = df[df['cod_est']==station_id].drop(columns=['cod_est']).reset_index(drop=True)
y = df_sta['ca_entradas']
X = df_sta.drop(columns=['ca_entradas'])

# 2. Selección preliminar de features por correlación
corr = X.corrwith(y).abs().sort_values(ascending=False)
keep = corr[corr>=corr_threshold]
X_sel = X[keep.index]

# Gráfico 1: correlaciones
plt.figure()
plt.barh(keep.index, keep.values)
plt.gca().invert_yaxis()
plt.title(f"Correlaciones |r| ≥ {corr_threshold} (Estación {station_id})")
plt.xlabel("|r|")
plt.tight_layout()
plt.show()

# 3. Configuración de CV temporal
tscv = TimeSeriesSplit(n_splits=cv_splits)

# 4. HalvingGridSearchCV
xgb = XGBRegressor(objective='reg:squarederror', tree_method='hist',
                   n_jobs=-1, random_state=random_state, verbosity=0)
search = HalvingGridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    factor=3,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    random_state=random_state,
    verbose=1,
    return_train_score=False
)

# 5. Búsqueda de hiperparámetros
search.fit(X_sel, y)

# 6. Gráfico 2: evolución de RMSE en Halving
results = pd.DataFrame(search.cv_results_)
best_per_iter = results.groupby('iter')['mean_test_score'].max()
plt.figure()
plt.plot(best_per_iter.index, -best_per_iter.values, marker='o')
plt.title("Evolución del RMSE (HalvingGridSearchCV)")
plt.xlabel("Iteración")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()

# 7. Evaluación CV manual y acumulación de predicciones
rmses, maes, r2s = [], [], []
residuals = []
y_pred_all = np.zeros_like(y, dtype=float)

for fold, (tr, te) in enumerate(tscv.split(X_sel), 1):
    model = search.best_estimator_.set_params(random_state=random_state)
    model.fit(X_sel.iloc[tr], y.iloc[tr],
              eval_set=[(X_sel.iloc[te], y.iloc[te])], verbose=False)
    y_pred = model.predict(X_sel.iloc[te])
    y_pred_all[te] = y_pred
    rmses.append(np.sqrt(mean_squared_error(y.iloc[te], y_pred)))
    maes.append(mean_absolute_error(y.iloc[te], y_pred))
    r2s.append(r2_score(y.iloc[te], y_pred))
    residuals.extend(y.iloc[te] - y_pred)

# 8. Gráfico 3: RMSE por fold
plt.figure()
plt.bar(range(1, cv_splits+1), rmses)
plt.title("RMSE por fold")
plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()

# 9. Gráfico 4: Predicción vs Real (CV completo)
plt.figure()
plt.scatter(y, y_pred_all, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--')
plt.title("Predicción vs Real (todos los folds)")
plt.xlabel("Real")
plt.ylabel("Predicción")
plt.tight_layout()
plt.show()

# 10. Test de normalidad de residuos y gráfico
stat, p_val = shapiro(residuals)
plt.figure()
plt.hist(residuals, bins=20, edgecolor='k')
plt.title(f"Histograma de residuos (Shapiro p={p_val:.3f})")
plt.xlabel("Residual")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# 11. Gráfico 5: importancias de features
feat_imp = pd.Series(search.best_estimator_.feature_importances_, index=keep.index)
feat_imp = feat_imp.sort_values(ascending=False)
plt.figure()
plt.barh(feat_imp.index, feat_imp.values)
plt.gca().invert_yaxis()
plt.title("Importancia de features (XGBoost)")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()

# 12. Métricas agregadas y parámetros óptimos
metrics = {
    'RMSE_mean': np.mean(rmses), 'RMSE_std': np.std(rmses),
    'MAE_mean': np.mean(maes),    'MAE_std': np.std(maes),
    'R2_mean': np.mean(r2s),      'R2_std': np.std(r2s)
}
print("\nMétricas CV:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
print("\nMejores parámetros:", search.best_params_)
