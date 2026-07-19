# generadorcsvreplicaanterior.py
# Genera summary_training.csv i summary_testing.csv
# Canvia DATASET per generar un o l'altre, o deixa-ho com està per generar els dos.

import os
import pandas as pd
from functools import reduce
from quantification_methods import get_acdc, build_metrics_dataframe

# CONFIGURACIÓ
BASE_PATH = r"C:\Users\Alex\Downloads\Projecte_estiu\ACDC\ACDC\database"
LABELS    = [1, 2, 3]  # 1=RV, 2=Miocardi, 3=LV

# Canvia a "training" o "testing" per generar només un,
# o deixa la llista per generar els dos
DATASETS  = ["training", "testing"]

# descriptors geometrics (sense intensitat/textura, veure nota als altres scripts)
features = [
    'centroid_radius', 'min_distance', 'max_distance','perimeter', 'area', 'circularity', 'eccentricity',
    'solidity', 'convexity', 'curvature', 'Elongation'
]


def process_dataset(dataset_split):
    print(f"\n{'='*60}")
    print(f"Processant: {dataset_split.upper()}")
    print(f"{'='*60}")

    # 1. Carregar dades
    acdc_data_path = os.path.join(BASE_PATH, dataset_split)
    acdc_data      = get_acdc(acdc_data_path)

    # 2. Calcular mètriques per cada label i fer la mitjana per pacient+fase
    # aqui NO filtro per fase, aixi ED i ES queden com files diferents dins el mateix csv
    dfs_summary = []
    for lbl in LABELS:
        print(f"\n  Calculant mètriques per label {lbl}...")
        df_lbl = build_metrics_dataframe(acdc_data, lbl)

        # Mitjana de les mètriques per pacient+fase (ED i ES separats)
        agg_dict = {f: 'mean' for f in features}
        agg_dict.update({'Height': 'first', 'Weight': 'first', 'Patient_group': 'first'})

        df_sum = (
            df_lbl
            .groupby('Patient_Information')[features + ['Height', 'Weight', 'Patient_group']]
            .agg(agg_dict)
            .reset_index()
        )

        # Afegir suffix per distingir labels (_L1, _L2, _L3)
        suffix = f"_L{lbl}"
        df_sum = df_sum.rename(columns={f: f"{f}{suffix}" for f in features})
        dfs_summary.append(df_sum)

    # 3. Merge dels tres labels
    df_merged = reduce(
        lambda left, right: pd.merge(
            left, right,
            on=['Patient_Information', 'Height', 'Weight', 'Patient_group'],
            how='outer'
        ),
        dfs_summary
    )

    # 4. Separar PatientID i Phase
    df_merged[['PatientID', 'Phase']] = (
        df_merged['Patient_Information']
        .str.rsplit('_', n=1, expand=True)
    )

    # 5. Reordenar columnes
    metrics_cols = [
        c for c in df_merged.columns
        if c not in ('Patient_Information', 'PatientID', 'Phase',
                     'Patient_group', 'Height', 'Weight')
    ]
    df_final = df_merged[
        ['PatientID', 'Phase']
        + metrics_cols
        + ['Patient_group', 'Height', 'Weight']
    ]

    # 6. Guardar CSV
    output_csv = f"summary_{dataset_split}.csv"
    df_final.to_csv(output_csv, index=False)
    print(f"\n  CSV guardat: {output_csv}")
    print(f"  Shape: {df_final.shape} ({len(df_final)} pacients x {len(df_final.columns)} columnes)")
    print(f"  Pacients: {df_final['PatientID'].nunique()}")
    print(f"  Fases: {df_final['Phase'].value_counts().to_dict()}")


def main():
    # genera el summary per cada dataset de la llista DATASETS
    for split in DATASETS:
        process_dataset(split)
    print(f"\n{'='*60}")
    print("Fet! CSVs generats correctament.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
