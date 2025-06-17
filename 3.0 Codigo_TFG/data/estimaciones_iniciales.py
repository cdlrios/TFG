import pandas as pd
import re
import numpy as np

# ============================================================
# RUTAS DE ARCHIVOS
# ============================================================
CUOTAS_FILE    = '/content/matriz_share_39dias.csv'
RED_FILE       = '/content/demanda_diaria.csv'
EVENTS_FILE    = '/content/eventos.csv'
CIERRE_FILE    = '/content/obras2024.csv'
OUTPUT_FILE    = '/content/estimacion_fest_20250611.csv'

# ============================================================
# 1) LECTURA Y NORMALIZACIÓN DEL CSV DE CUOTAS
# ============================================================
_raw = pd.read_csv(CUOTAS_FILE, sep=';')
raw  = _raw.loc[:, ~_raw.columns.str.contains('^Unnamed')]
if {'cod_est','fecha'}.issubset(raw.columns):
    cuotas = raw.rename(columns={'cuota_entradas':'cuota'})
    cuotas['fecha'] = pd.to_datetime(cuotas['fecha'])
else:
    date_cols = [c for c in raw.columns if re.match(r"\d{4}-\d{2}-\d{2}", str(c))]
    cuotas = (
        raw.melt(
            id_vars='cod_est',
            value_vars=date_cols,
            var_name='fecha',
            value_name='cuota'
        )
        .assign(fecha=lambda df: pd.to_datetime(df['fecha']))
    )

# ============================================================
# 2) LECTURA Y NORMALIZACIÓN DEL CSV DE EVENTOS
# ============================================================
events = (
    pd.read_csv(EVENTS_FILE, sep=';', parse_dates=['fecha'])
      .rename(columns={'coeficiente_asistencia':'coef'})
)[['cod_est','fecha','coef']].drop_duplicates()
events['coef'] = (
    events['coef'].astype(str)
          .str.replace(',','.',regex=False)
)
events['coef'] = pd.to_numeric(events['coef'], errors='coerce').fillna(0)

# Calcular dow y quarter para todo el periodo (medias globales)
cuotas['dow'] = cuotas['fecha'].dt.dayofweek
cuotas['quarter'] = cuotas['fecha'].dt.quarter

# Identificar estaciones con recinto (al menos un evento)
recinto_ids = events.loc[events['coef']>0, 'cod_est'].unique()

# ============================================================
# 3) PREPARACIÓN DE DATOS PARA ESTACIONES CON RECINTO
# ============================================================
cuotas_recinto = cuotas[cuotas['cod_est'].isin(recinto_ids)]
cuotas_recinto = (
    cuotas_recinto
      .merge(events[['cod_est','fecha']], on=['cod_est','fecha'], how='left', indicator=True)
      .query("_merge=='left_only'")
      .drop(columns=['_merge'])
)
cuotas_recinto['dow']     = cuotas_recinto['fecha'].dt.dayofweek
cuotas_recinto['quarter'] = cuotas_recinto['fecha'].dt.quarter

# Calcular promedios sin días de evento
prom_qtr_rec     = cuotas_recinto.groupby(['cod_est','dow','quarter'], as_index=False)['cuota'].mean()
fallback_dow_rec = cuotas_recinto.groupby(['cod_est','dow'],           as_index=False)['cuota'].mean().rename(columns={'cuota':'cuota_dow'})
fallback_est_rec = cuotas_recinto.groupby(['cod_est'],                  as_index=False)['cuota'].mean().rename(columns={'cuota':'cuota_est'})

# ============================================================
# 4) LECTURA DE LA DEMANDA TOTAL DE LA RED
# ============================================================
red = (
    pd.read_csv(RED_FILE, sep=';', parse_dates=['fecha'])
      .rename(columns={'entradas':'entradas_red'})
)

# ============================================================
# 5) CREACIÓN DE LA MALLA ESTACIÓN × DÍA Y TRATAMIENTO DE FESTIVOS
# ============================================================
calendar = pd.date_range('2024-01-01','2024-12-31',freq='D').to_frame(index=False,name='fecha')
calendar['dow']     = calendar['fecha'].dt.dayofweek
calendar['quarter']= calendar['fecha'].dt.quarter
festivos = pd.to_datetime([
    '2024-01-01','2024-03-28','2024-03-29',
    '2024-05-01','2024-05-02','2024-07-25',
    '2024-08-15','2024-12-06','2024-12-25'
])
calendar['dow'] = np.where(calendar['fecha'].isin(festivos), 6, calendar['dow'])
stations = cuotas[['cod_est']].drop_duplicates()
skeleton = calendar.assign(key=1).merge(stations.assign(key=1), on='key').drop(columns='key')
skeleton['has_recinto'] = skeleton['cod_est'].isin(recinto_ids)

# ============================================================
# 6) FUNCIONES UTILES
# ============================================================

# Pipeline completo de estimación (calibración X, evento, cierres)
def build_full_pipeline(sk, prom_qtr, fallback_dow, fallback_est):
    # Estimación base
    df = (sk
        .merge(prom_qtr,     on=['cod_est','dow','quarter'], how='left')
        .merge(fallback_dow, on=['cod_est','dow'],            how='left')
        .merge(fallback_est, on=['cod_est'],                  how='left')
        .assign(cuota_final=lambda d: d['cuota'].fillna(d['cuota_dow']).fillna(d['cuota_est']))
        .merge(red, on='fecha', how='left')
        .assign(entradas_est=lambda d: d['cuota_final'] * d['entradas_red'])
    )
    # Entradas reales para calibración
    entradas_reales = (
        cuotas
          .merge(red, on='fecha', how='left')
          .assign(entradas_real=lambda d: d['cuota'] * d['entradas_red'])
          [['cod_est','fecha','entradas_real']]
    )
    # Calibración X (solo días con evento)
    _calib = (df
        .merge(entradas_reales, on=['cod_est','fecha'], how='inner')
        .merge(events[['cod_est','fecha','coef']],    on=['cod_est','fecha'], how='left')
        .assign(delta=lambda d: d['entradas_real'] - d['entradas_est'])
    )
    calib = (_calib.loc[_calib['coef'] > 0]
        .assign(X=lambda d: d['delta'] / d['coef'])
        .groupby('cod_est', as_index=False)['X'].mean()
    )
    global_X = calib['X'].mean()
    global_X = 0 if pd.isna(global_X) else global_X
    # Ajuste por evento
    df = (df
        .merge(events[['cod_est','fecha','coef']], on=['cod_est','fecha'], how='left')
        .merge(calib, on='cod_est', how='left')
        .assign(coef=lambda d: d['coef'].fillna(0), X=lambda d: d['X'].fillna(global_X))
        .assign(entradas_est_evento=lambda d: d['entradas_est'] + d['coef'] * d['X'])
        .merge(entradas_reales, on=['cod_est','fecha'], how='left')
        .assign(entradas=lambda d: d['entradas_real'].fillna(d['entradas_est_evento']))
    )
    # Ajuste por cierres
    cierres = (
        pd.read_csv(CIERRE_FILE, sep=';', parse_dates=['fecha'])
           .rename(columns={'cof_cierre':'coef_cierre'})
          [['cod_est','fecha','coef_cierre']].drop_duplicates()
    )
    cierres['coef_cierre'] = (
        cierres['coef_cierre'].astype(str).str.replace(',','.',regex=False)
    )
    cierres['coef_cierre'] = pd.to_numeric(cierres['coef_cierre'], errors='coerce').fillna(0)
    df = df.merge(cierres, on=['cod_est','fecha'], how='left')
    df['coef_cierre'] = df['coef_cierre'].fillna(0)
    df['entradas'] = (df['entradas'].fillna(0) * (1 - df['coef_cierre'])).round().astype(int)
    return df[['cod_est','fecha','entradas']]

# ============================================================
# 7) EJECUCIÓN SEGREGADA POR TIPO DE ESTACIÓN
# ============================================================
# Con recinto: usar medias sin eventos
sk_recinto    = skeleton[skeleton['has_recinto']]
df_recinto    = build_full_pipeline(sk_recinto, prom_qtr_rec, fallback_dow_rec, fallback_est_rec)
# Sin recinto: usar medias completas (usamos las originales calculadas sobre 'cuotas'):
prom_qtr_all     = cuotas.groupby(['cod_est','dow','quarter'], as_index=False)['cuota'].mean()
fallback_dow_all = cuotas.groupby(['cod_est','dow'],         as_index=False)['cuota'].mean().rename(columns={'cuota':'cuota_dow'})
fallback_est_all = cuotas.groupby(['cod_est'],                as_index=False)['cuota'].mean().rename(columns={'cuota':'cuota_est'})
sk_no_recinto = skeleton[~skeleton['has_recinto']]
df_no_recinto = build_full_pipeline(sk_no_recinto, prom_qtr_all, fallback_dow_all, fallback_est_all)

# ============================================================
# 8) UNIÓN Y EXPORTACIÓN FINAL
# ============================================================
df_all = pd.concat([df_recinto, df_no_recinto], ignore_index=True)
df_all.to_csv(OUTPUT_FILE, index=False)
print(f"✔ CSV generado segregado recintos/no_recintos en {OUTPUT_FILE}")
