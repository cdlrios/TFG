import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import shapiro
import numpy as np
from xgboost import XGBRegressor
from tqdm import tqdm
import tqdm_joblib


def evaluate_model_cv(est, df_sta, date_col=None,
                      n_splits=5, random_state=42):
    """
    Realiza búsqueda rápida de hiperparámetros con CV anidada ligera y early stopping.
    Incluye progreso con tqdm.
    """
    # Separar X, y
    y = df_sta['ca_entradas']
    X = df_sta.drop(columns=['ca_entradas'])

    # Configurar splitter temporal o clásico
    if date_col and date_col in X.columns:
        X = X.sort_values(date_col)
        y = y.loc[X.index]
        cv_split = TimeSeriesSplit(n_splits=n_splits)
        X = X.drop(columns=[date_col])
    else:
        cv_split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Modelo base
    base_model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        random_state=random_state,
        verbosity=0,           # uso de verbosity en lugar de verbose
        early_stopping_rounds=10,
        eval_metric='rmse'
    )

    # Espacio de búsqueda reducido
    param_dist = {
        'n_estimators': [100, 300, 600],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 8]
    }

    # Búsqueda halving para acelerar
    search = HalvingRandomSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        factor=3,
        max_resources='auto',
        min_resources='smallest',
        aggressive_elimination=True,
        random_state=random_state,
        cv=cv_split,
        n_jobs=1,
        scoring='neg_root_mean_squared_error',
        verbose=0
    )

    rmse_scores, r2_scores, mae_scores, residuals = [], [], [], []

    # CV externo con progreso
    for train_idx, test_idx in tqdm(cv_split.split(X, y), total=n_splits,
                                     desc=f'Estación {est}', leave=False):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # Ajuste con early stopping usando eval_set
        search.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=False
        )
        best = search.best_estimator_

        y_pred = best.predict(X_te)

        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2 = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)

        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mae_scores.append(mae)
        residuals.extend(y_te - y_pred)

    stats = {
        'cod_est': est,
        'R2_mean': np.mean(r2_scores),
        'R2_std': np.std(r2_scores),
        'MAE_mean': np.mean(mae_scores),
        'MAE_std': np.std(mae_scores),
        'RMSE_mean': np.mean(rmse_scores),
        'RMSE_std': np.std(rmse_scores),
        'n_feat': X.shape[1]
    }

    stat, p_val = shapiro(residuals)
    stats.update({'res_shapiro_stat': stat, 'res_shapiro_p': p_val})

    return stats


def main():
    df = pd.read_csv(Path('data/dataset_metro.csv'), sep=';', decimal=',')
    date_col = 'fecha' if 'fecha' in df.columns else None
    station_dfs = {
        est: data.drop(columns=['cod_est']).reset_index(drop=True)
        for est, data in df.groupby('cod_est')
    }

    with tqdm_joblib.tqdm_joblib(tqdm(desc='Procesando estaciones', total=len(station_dfs))):
        results = Parallel(n_jobs=6, backend='threading')(
            delayed(evaluate_model_cv)(est, df_sta, date_col)
            for est, df_sta in station_dfs.items()
        )

    pd.DataFrame(results).to_csv('xgb_summary_cv_fast.csv', index=False)
    print('✅ Resultados guardados en xgb_summary_cv_fast.csv')


if __name__ == '__main__':
    main()
