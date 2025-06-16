
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint

# Import experimental halving search
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import (
    KFold,
    TimeSeriesSplit,
    HalvingRandomSearchCV
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_data(path: Path) -> pd.DataFrame:
    """
    Carga el dataset desde CSV con separador ';' y coma decimal.
    """
    df = pd.read_csv(path, sep=";", decimal=",")
    logging.info(f"Dataset cargado con {len(df)} registros.")
    return df


def select_features_by_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float
) -> pd.DataFrame:
    """
    Filtra columnas de X cuya correlación absoluta con y >= threshold.
    """
    corr = X.corrwith(y).abs()
    selected = corr[corr >= threshold].index.tolist()
    logging.info(f"Seleccionadas {len(selected)}/{X.shape[1]} features con |r|>={threshold}.")
    return X[selected]


def run_rf_for_station(
    est_code: str,
    df_sta: pd.DataFrame,
    corr_threshold: float,
    cv_folds: int,
    use_timeseries: bool,
    param_dist: dict,
    n_iter_search: int,
    random_state: int
) -> (dict, pd.DataFrame):
    """
    Entrena y valida un Random Forest para una estación usando HalvingRandomSearchCV.
    Devuelve diccionario de métricas y DataFrame de importancias.
    """
    y = df_sta["ca_entradas"].values
    X = df_sta.drop(columns=["ca_entradas"])

    # Selección opcional de features
    if corr_threshold > 0:
        X = select_features_by_correlation(X, pd.Series(y), corr_threshold)

    # Definir esquema de CV
    if use_timeseries:
        cv = TimeSeriesSplit(n_splits=cv_folds)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Configurar búsqueda halving de hiperparámetros
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    search = HalvingRandomSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        factor=3,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )
    logging.info(f"[{est_code}] Iniciando HalvingRandomSearchCV con factor=3 y {cv_folds}-CV.")
    search.fit(X, y)
    best = search.best_estimator_

    # Validación cruzada final con cálculo manual de RMSE
    rmses, maes, r2s = [], [], []
    for train_idx, test_idx in cv.split(X):
        best.fit(X.iloc[train_idx], y[train_idx])
        y_pred = best.predict(X.iloc[test_idx])
        mse_val = mean_squared_error(y[test_idx], y_pred)
        rmses.append(np.sqrt(mse_val))
        maes.append(mean_absolute_error(y[test_idx], y_pred))
        r2s.append(r2_score(y[test_idx], y_pred))
    metrics = {
        'cod_est': est_code,
        'n_features': X.shape[1],
        'mean_RMSE': np.mean(rmses),
        'std_RMSE': np.std(rmses),
        'mean_MAE': np.mean(maes),
        'std_MAE': np.std(maes),
        'mean_R2': np.mean(r2s),
        'std_R2': np.std(r2s),
        'best_params': search.best_params_
    }

    # Importancias de features
    feat_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': best.feature_importances_
    }).sort_values('importance', ascending=False)

    return metrics, feat_imp


def run_rf_all_stations(
    station_dfs: dict,
    corr_threshold: float,
    cv_folds: int,
    use_timeseries: bool,
    param_dist: dict,
    n_iter_search: int,
    random_state: int
) -> (pd.DataFrame, pd.DataFrame):
    """
    Ejecuta run_rf_for_station para cada estación y concatena resultados.
    """
    summaries, feats = [], []
    for est, df_sta in station_dfs.items():
        try:
            metrics, feat_imp = run_rf_for_station(
                est, df_sta, corr_threshold, cv_folds,
                use_timeseries, param_dist, n_iter_search, random_state
            )
            summaries.append(metrics)
            feat_imp['cod_est'] = est
            feats.append(feat_imp)
        except Exception as e:
            logging.error(f"Error en estación {est}: {e}")
            summaries.append({'cod_est': est, 'error': str(e)})
    return pd.DataFrame(summaries), pd.concat(feats, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena RF con validación y búsqueda de hiperparámetros por halving para cada estación."
    )
    parser.add_argument(
        '-i', '--input', type=Path, default=Path('data/dataset_metro.csv'),
        help='Ruta al CSV con datos por estación'
    )
    parser.add_argument(
        '-o', '--output', type=Path, default=Path('rf_summary_all_stations.csv'),
        help='Fichero de salida para métricas'
    )
    parser.add_argument(
        '--feat-out', type=Path, default=Path('feature_importances_all_stations.csv'),
        help='Fichero de salida para importancias de features'
    )
    parser.add_argument(
        '--corr-threshold', type=float, default=0.05,
        help='Umbral mínimo de correlación absoluta para selección de features'
    )
    parser.add_argument('--cv-folds', type=int, default=5, help='Número de folds para CV')
    parser.add_argument(
        '--time-series', action='store_true',
        help='Usar TimeSeriesSplit en lugar de KFold'
    )
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    df = load_data(args.input)
    station_dfs = {
        est: data.drop(columns=['cod_est']).reset_index(drop=True)
        for est, data in df.groupby('cod_est')
    }

    param_dist = {
        'n_estimators': randint(100, 601),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', None]
    }

    summary_df, feats_df = run_rf_all_stations(
        station_dfs,
        corr_threshold=args.corr_threshold,
        cv_folds=args.cv_folds,
        use_timeseries=args.time_series,
        param_dist=param_dist,
        n_iter_search=20,
        random_state=42
    )
    summary_df.to_csv(args.output, index=False)
    feats_df.to_csv(args.feat_out, index=False)
    logging.info(f"Guardado resumen en {args.output}")
    logging.info(f"Guardado importancias en {args.feat_out}")


if __name__ == '__main__':
    main()
