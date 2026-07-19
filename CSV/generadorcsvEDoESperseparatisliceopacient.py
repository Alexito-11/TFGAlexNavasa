# generadorcsvEDoESperseparatisliceopacient.py
# genera els csv "summary" (per slice i per pacient) amb els descriptors geometrics
# manuals del script quantification_methods, per una fase (ED o ES) i per label (1,2,3)

import os
import pandas as pd
from functools import reduce
from quantification_methods import get_acdc, build_metrics_dataframe

# CONFIGURACIÓ
OUTPUT_DIR = r"C:\Users\Alex\PycharmProjects\PythonProject"  # ← carpeta on guardar
BASE_PATH  = r"C:\Users\Alex\Downloads\Projecte_estiu\ACDC\ACDC\database"
LABELS     = [1, 2, 3]       # 1=RV, 2=Miocardi, 3=LV
DATASETS   = ["training", "testing"]
PHASE      = "ES"            # ← Canvia a "ED" o "ES"

# Descriptors geomètrics. S'han eliminat contrast, Homogeneity, Mean, Median, STD,
# Max_Intensity i Min_Intensity perquè amb el GT com a imatge són constants per
# construcció (tots els píxels d'una estructura tenen el mateix valor 1, 2 o 3).
features = [
    'centroid_radius', 'min_distance', 'max_distance',
    'perimeter', 'area', 'circularity', 'eccentricity',
    'solidity', 'convexity', 'curvature', 'Elongation'
]


def process_dataset(dataset_split):
    print(f"\n{'='*60}")
    print(f"Processant: {dataset_split.upper()} — Fase: {PHASE}")
    print(f"{'='*60}")

    acdc_data_path = os.path.join(BASE_PATH, dataset_split)
    acdc_data      = get_acdc(acdc_data_path)

    dfs_slice   = []
    dfs_summary = []

    for lbl in LABELS:
        print(f"\n  Calculant mètriques per label {lbl}...")
        df_lbl = build_metrics_dataframe(acdc_data, lbl)

        # nomes em quedo amb les files de la fase que toca (ED o ES)
        df_lbl = df_lbl[df_lbl['Patient_Information'].str.endswith(f'_{PHASE}')]

        if df_lbl.empty:
            print(f"  ATENCIÓ: cap dada per label {lbl} i fase {PHASE}")
            continue

        # afegeixo sufix _L{lbl} a cada feature perque cada label tingui columnes separades
        suffix = f"_L{lbl}"
        df_lbl_renamed = df_lbl.rename(columns={f: f"{f}{suffix}" for f in features})
        dfs_slice.append(df_lbl_renamed)

        # per l'agregat per pacient, faig mitjana de cada feature entre totes les slices
        agg_dict = {f: 'mean' for f in features}
        agg_dict.update({'Height': 'first', 'Weight': 'first', 'Patient_group': 'first'})

        df_sum = (
            df_lbl
            .groupby('Patient_Information')[features + ['Height', 'Weight', 'Patient_group']]
            .agg(agg_dict)
            .reset_index()
        )
        df_sum = df_sum.rename(columns={f: f"{f}{suffix}" for f in features})
        dfs_summary.append(df_sum)

    if dfs_slice:
        # ajunto els 3 labels (RV, Miocardi, LV) en un unic df, una fila per slice
        df_slice_merged = reduce(
            lambda left, right: pd.merge(
                left, right,
                on=['Patient_Information', 'Slice_No', 'Height', 'Weight', 'Patient_group'],
                how='outer'
            ),
            dfs_slice
        )
        # separo PatientID i Phase de la columna combinada Patient_Information
        df_slice_merged[['PatientID', 'Phase']] = (
            df_slice_merged['Patient_Information']
            .str.rsplit('_', n=1, expand=True)
        )
        metrics_slice_cols = [c for c in df_slice_merged.columns
                              if c not in ('Patient_Information', 'PatientID', 'Phase',
                                           'Slice_No', 'Patient_group', 'Height', 'Weight')]
        df_slice_final = df_slice_merged[
            ['PatientID', 'Phase', 'Slice_No']
            + metrics_slice_cols
            + ['Patient_group', 'Height', 'Weight']
        ]
        csv_slice = os.path.join(OUTPUT_DIR, f"summary_{dataset_split}_{PHASE}_per_slice.csv")
        df_slice_final.to_csv(csv_slice, index=False)
        print(f"\n  CSV per slice guardat: {csv_slice}")
        print(f"  Shape: {df_slice_final.shape}")

    if dfs_summary:
        # mateix procediment pero pel csv agregat a nivell de pacient
        df_merged = reduce(
            lambda left, right: pd.merge(
                left, right,
                on=['Patient_Information', 'Height', 'Weight', 'Patient_group'],
                how='outer'
            ),
            dfs_summary
        )
        df_merged[['PatientID', 'Phase']] = (
            df_merged['Patient_Information']
            .str.rsplit('_', n=1, expand=True)
        )
        metrics_cols = [c for c in df_merged.columns
                        if c not in ('Patient_Information', 'PatientID', 'Phase',
                                     'Patient_group', 'Height', 'Weight')]
        df_final = df_merged[
            ['PatientID', 'Phase']
            + metrics_cols
            + ['Patient_group', 'Height', 'Weight']
        ]
        csv_pacient = os.path.join(OUTPUT_DIR, f"summary_{dataset_split}_{PHASE}_per_pacient.csv")
        df_final.to_csv(csv_pacient, index=False)
        print(f"\n  CSV per pacient guardat: {csv_pacient}")
        print(f"  Shape: {df_final.shape}")
        print(f"  Pacients: {df_final['PatientID'].nunique()}")


def main():
    # processa training i testing per la fase configurada a PHASE
    for split in DATASETS:
        process_dataset(split)
    print(f"\n{'='*60}")
    print(f"Fet! CSVs generats per fase {PHASE}.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
