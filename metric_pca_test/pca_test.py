import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

output_dir = 'metric_pca_test'

# Robusztusabb szűrés: csak a konkrétan '_flow' részt tartalmazó fájlokat vesszük fel
csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and '_flow' in f]

if not csv_files:
    print(f"Hiba: Nem találhatók megfelelő '_flow' tartalmú adatfájlok a '{output_dir}' mappában.")
    exit(1)

all_data = []
for csv_file in csv_files:
    try:
        df = pd.read_csv(os.path.join(output_dir, csv_file))
        flow_level = int(csv_file.split('_flow')[1].split('_')[0])
        df['flow_level'] = flow_level
        all_data.append(df)
    except Exception as e:
        print(f"Figyelmeztetés: A {csv_file} feldolgozása sikertelen, a hiba oka: {e}")

if not all_data:
    print("Hiba: Egyetlen érvényes adatfájlt sem sikerült feldolgozni.")
    exit(1)

full_df = pd.concat(all_data, ignore_index=True)
print(f'Osszes adatpont: {len(full_df)}')

metric_cols = ['TotalTravelTime', 'AvgTravelTime', 'TotalWaitingTime',
               'AvgWaitingTime', 'TotalCO2', 'AvgCO2', 'VehCount',
               'AvgSpeed', 'AvgOccupancy', 'QueueLength']

mask = full_df['VehCount'] > 0
df_valid = full_df[mask].copy()
print(f'Ervenyes adatpont: {len(df_valid)}')

if df_valid.empty:
    print("Hiba: Az érvényes adatok száma nulla, a PCA elemzés nem futtatható.")
    exit(1)

epsilon = 1e-5
df_log = np.log(df_valid[metric_cols].clip(lower=epsilon) + epsilon)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_log)
pca = PCA()
pca_result = pca.fit_transform(data_scaled)
explained_var = pca.explained_variance_ratio_ * 100
components = pca.components_

print('\n--- LOADINGS (PC1, PC2, PC3) ---')
loadings_df = pd.DataFrame(components[:3].T, columns=['PC1', 'PC2', 'PC3'], index=metric_cols)
print(loadings_df.round(4))

print(f'\n--- EXPLAINED VARIANCE ---')
for i, var in enumerate(explained_var[:5]):
    print(f'  PC{i+1}: {var:.2f}%  (cumulative: {sum(explained_var[:i+1]):.2f}%)')

print('\n--- NORMALIZACIOS PARAMETEREK (log-sigmoid) ---')
for col in metric_cols:
    vals = df_valid[col].values
    vals = vals[vals > 0]
    if len(vals) > 0:
        log_vals = np.log(vals + 1e-5)
        mu = np.mean(log_vals)
        std = np.std(log_vals)
        print(f'  {col:<20} MU={mu:10.4f}  STD={std:8.4f}  (raw: median={np.median(vals):.2f}, p5={np.percentile(vals,5):.2f}, p95={np.percentile(vals,95):.2f})')
