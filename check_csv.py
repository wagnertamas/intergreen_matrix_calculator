import os
import pandas as pd

dir_path = 'metric_pca_per_junction'
files = sorted([f for f in os.listdir(dir_path) if f.endswith('.csv') and '_flow' in f])
broken = []

for f in files:
    full_path = os.path.join(dir_path, f)
    try:
        pd.read_csv(full_path)
    except Exception as e:
        print(f"BROKEN: {full_path} - {e}")
        broken.append(full_path)

if not broken:
    print("No broken files found.")
else:
    print(f"Found {len(broken)} broken files.")
