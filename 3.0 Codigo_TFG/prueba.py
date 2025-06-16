#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrenamiento de Random Forest para predicción de tráfico de Metro por estación
con validación temporal **sin fuga de información**.

Version 2 (15 jun 2025)
---------------------------------------------------------
Cambios clave respecto a la versión inicial:
* El filtrado correlacional de variables se realiza **dentro de cada fold**
  (solo con datos del pasado) evitando *data‑leakage*.
* La importancia de variables se **promedia** a lo largo de los folds para
  estabilizar el ranking.
* Se añaden pequeñas mejoras de reproducibilidad y estilo (argparse help,
  tipado, seed consistente).

Salida (carpeta outputs/):
- rf_summary.csv         → métricas medias ± desviación y parámetros finales
- var_imp_<cod_est>.csv  → ranking de importancia de variables por estación
- meta.json              → versiones de librerías, seed y parámetros de ejecución

Ejemplo de uso:
    python rf_time_series_cv_noleak.py \
        --csv_path data/dataset_metro.csv \
        --min_abs_r 0.05 \
        --n_splits 3 \
        --rf_iters 10
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.utils import check_random_state

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn.ensemble")

# Parámetros globales y constantes
MES_COL: str = "nu_mes"
DIA_COL: str = "nu_dia_mes"
OBJ_COL: str = "ca_entradas"
EST_COL: str = "cod_est"
SEED: int = 42
N_JOBS: int = max(mp.cpu_count() - 1, 1)
MIN_R: float = 0.05
MIN_VARS: int = 3

# Espacio de búsqueda de hiperparámetros
RF_PARAMS: Dict[str, List[Any]] = {
    "n_estimators": np.arange(200, 1001, 100).tolist(),
    "max_depth": [None, 5, 10, 20],
    "min_samples_leaf": [2, 4],
    "max_features": ["sqrt", 0.5],
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # helper para legibilidad
    """Root‑Mean‑Squared‑Error como float nativo."""
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def procesar_estacion(
    estacion: str,
    df_est: pd.DataFrame,
    min_r: float,
    min_vars: int,
    n_splits: int,
    rf_iters: int,
    seed: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Entrena y valida un modelo por estación evitando fuga de información."""

    print(f"▶ Estación {estacion}")

    # Orden temporal explícito (por si el CSV llega desordenado)
    df_est = df_est.sort_values([MES_COL, DIA_COL]).reset_index(drop=True)
    y = df_est[OBJ_COL]
    X = df_est.drop(columns=[OBJ_COL])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rf_base = RandomForestRegressor(random_state=seed, n_jobs=-1)

    modelos: List[RandomForestRegressor] = []
    metricas: List[Dict[str, float]] = []
    imp_acum: defaultdict[str, List[float]] = defaultdict(list)

    rng = check_random_state(seed)

    for i, (tr, te) in enumerate(tscv.split(X), 1):
        print(f"  - Fold {i}/{n_splits}")

        # 👉 Filtrado de variables SOLO con el subconjunto de entrenamiento
        corr = X.iloc[tr].corrwith(y.iloc[tr]).abs()
        sel = corr[corr >= min_r].index.tolist()
        if len(sel) < min_vars:
            sel = corr.nlargest(min_vars).index.tolist()

        X_tr, X_te = X.iloc[tr][sel], X.iloc[te][sel]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        busq = RandomizedSearchCV(
            rf_base,
            param_distributions=RF_PARAMS,
            n_iter=rf_iters,
            cv=2,
            scoring="neg_root_mean_squared_error",
            random_state=rng.randint(100000),
            n_jobs=1,
            verbose=0,
        )
        busq.fit(X_tr, y_tr)
        rf_best: RandomForestRegressor = busq.best_estimator_
        modelos.append(rf_best)

        # Métricas
        y_pred = rf_best.predict(X_te)
        metricas.append(
            {
                "RMSE": rmse(y_te.to_numpy(), y_pred),
                "MAE": mean_absolute_error(y_te, y_pred),
                "R2": r2_score(y_te, y_pred),
            }
        )

        # Importancia de variables
        for var, imp in zip(sel, rf_best.feature_importances_):
            imp_acum[var].append(float(imp))

    # Resumen de métricas
    df_m = pd.DataFrame(metricas)
    resumen: Dict[str, Any] = {
        "cod_est": estacion,
        "n_muestras": len(df_est),
        "n_vars_prom": np.mean([len(v) for v in imp_acum.values()]),
    }
    for col in ["RMSE", "MAE", "R2"]:
        resumen[f"{col}_mean"] = df_m[col].mean()
        resumen[f"{col}_std"] = df_m[col].std(ddof=1)

    # Parámetros finales del último modelo (para inspección rápida)
    params = modelos[-1].get_params()
    resumen.update(
        {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "min_samples_leaf": params["min_samples_leaf"],
            "max_features": params["max_features"],
        }
    )

    # Promedio de importancia de variables
    df_imp = (
        pd.DataFrame(
            {
                "variable": list(imp_acum.keys()),
                "importancia": [np.mean(v) for v in imp_acum.values()],
            }
        )
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )

    return resumen, df_imp


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--csv_path", default="data/dataset_metro.csv")
    parser.add_argument("--min_abs_r", type=float, default=MIN_R)
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--rf_iters", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=N_JOBS)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, sep=";", decimal=",")
    required_cols = {MES_COL, DIA_COL, EST_COL, OBJ_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    datos = {
        est: g.drop(columns=[EST_COL]).reset_index(drop=True)
        for est, g in df.groupby(EST_COL)
    }

    print(f"Procesando {len(datos)} estaciones en paralelo…")
    res = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(procesar_estacion)(
            est,
            d,
            args.min_abs_r,
            MIN_VARS,
            args.n_splits,
            args.rf_iters,
            SEED,
        )
        for est, d in datos.items()
    )

    resu, imps = zip(*res)
    df_res = pd.DataFrame(resu).sort_values("RMSE_mean")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    df_res.to_csv(out_dir / "rf_summary.csv", index=False)

    # Guardar importancias y último modelo por estación
    for (est, df_i), (_, mdl) in zip(imps, res):
        df_i.to_csv(out_dir / f"var_imp_{est}.csv", index=False)
        dump(mdl, out_dir / f"rf_model_{est}.joblib")  # opcional

    meta = {
        "python": platform.python_version(),
        "args": vars(args),
        "seed": SEED,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print("✅ Listo. Resultados en outputs/")


if __name__ == "__main__":
    main()