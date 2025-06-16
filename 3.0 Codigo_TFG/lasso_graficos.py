import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn

# ===============================
# Script Lasso con CV y reproducibilidad
# ===============================

# Semilla fija para reproducibilidad
np.random.seed(42)

# Mostrar versiones de librerías para documentación
print(f"pandas {pd.__version__}, numpy {np.__version__}, scikit-learn {sklearn.__version__}")

# Parámetros
file_path = Path("data/dataset_metro.csv")
station_id = 708 
min_abs_r = 0.05

# 1. Carga y filtrado de datos para la estación
df = pd.read_csv(file_path, sep=";", decimal=",")
df_sta = df[df["cod_est"] == station_id].drop(columns=["cod_est"]).reset_index(drop=True)
y = df_sta["ca_entradas"]
X = df_sta.drop(columns=["ca_entradas"])

# 2. Selección preliminar por correlación
corr = X.corrwith(y).abs()
keep_cols = corr[corr >= min_abs_r].sort_values(ascending=False).index.tolist()
X_sel = X[keep_cols]

# Gráfico 1: Correlaciones absolutas
plt.figure()
plt.barh(keep_cols, corr[keep_cols])
plt.title(f"Correlaciones |r| ≥ {min_abs_r} con 'ca_entradas' (Estación {station_id})")
plt.xlabel("Valor absoluto de r")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 3. División Train/Test (con semilla)
X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.2, random_state=42
)

# 4. Pipeline: escalado y LassoCV (CV interno)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(cv=5, random_state=42, max_iter=10000))
])
pipe.fit(X_train, y_train)
lasso = pipe.named_steps["lasso"]

# 5. Evolución del MSE durante la validación cruzada
mse_path = lasso.mse_path_  # shape: (n_alphas, n_folds)
mean_mse = np.mean(mse_path, axis=1)
plt.figure()
plt.plot(lasso.alphas_, mean_mse)
plt.xscale('log')
plt.xlabel("α (log scale)")
plt.ylabel("Mean MSE across folds")
plt.title("Evolución del MSE en función de α (LassoCV)")
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()

# 6. Gráfico de coeficientes no nulos
coef = pd.Series(lasso.coef_, index=keep_cols)
coef_nonzero = coef[coef != 0]
plt.figure()
plt.barh(coef_nonzero.index, coef_nonzero.values)
plt.title(f"Coeficientes Lasso (no nulos) – α óptimo={lasso.alpha_:.4f}")
plt.xlabel("Coeficiente")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 7. Predicción vs Real
y_pred = pipe.predict(X_test)
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--')
plt.title("Predicción vs Valores Reales")
plt.xlabel("Real (y_test)")
plt.ylabel("Predicción")
plt.tight_layout()
plt.show()

# 8. Impresión de métricas
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
mse = mean_squared_error(y_test, y_pred)
metrics = {
    "R2": r2_score(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": np.sqrt(mse),
    "n_features_used": coef_nonzero.shape[0],
    "alpha_opt": lasso.alpha_
}
print("\nMétricas de evaluación:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
