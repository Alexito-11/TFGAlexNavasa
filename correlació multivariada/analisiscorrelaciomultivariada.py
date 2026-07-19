# multivariate_analysis.py
# Anàlisi de correlació MULTIVARIADA: prediu cada feature manual
# a partir de les 256 dimensions 'interm' combinades, usant Random Forest.
#
# Es processen automàticament totes les combinacions:
#   dataset  ∈ {training, testing}
#   phase    ∈ {ED, ES}
#   level    ∈ {per_slice, per_pacient}
#
# IMPORTANT: a nivell 'per_slice' s'usa GroupKFold per PatientID,
# per evitar que slices del mateix pacient quedin repartides entre
# train/test del cross-validation (data leakage).

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # backend sense interfície gràfica
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, GroupKFold

# CONFIGURACIÓ
INPUT_DIR  = r"C:\Users\Alex\PycharmProjects\PythonProject"
OUTPUT_DIR = r"C:\Users\Alex\PycharmProjects\PythonProject\correlacions_multivariades"

DATASETS = ["training", "testing"]
PHASES   = ["ED", "ES"]
LEVELS   = ["per_slice", "per_pacient"]

N_ESTIMATORS = 100
N_SPLITS     = 5
RANDOM_STATE = 42

# columnes que no son features (metadata del pacient), per no ficar-les al model
NON_FEATURE_COLS = {
    'PatientID', 'Phase', 'Slice_No', 'Label',
    'Patient_group', 'Height', 'Weight', 'Patient_Information'
}


def load_pair(dataset, phase, level):
    # carrego els dos csv (features manuals i features interm de la xarxa) i els junto
    manual_csv = os.path.join(INPUT_DIR, f"summary_{dataset}_{phase}_{level}.csv")
    interm_csv = os.path.join(INPUT_DIR, f"interm_{dataset}_{phase}_{level}.csv")

    if not os.path.exists(manual_csv):
        print(f"  [SKIP] No trobat: {manual_csv}")
        return None
    if not os.path.exists(interm_csv):
        print(f"  [SKIP] No trobat: {interm_csv}")
        return None

    df_manual = pd.read_csv(manual_csv)
    df_interm = pd.read_csv(interm_csv)

    merge_keys = ['PatientID', 'Phase']
    if level == "per_slice":
        merge_keys.append('Slice_No')

    merged = pd.merge(df_manual, df_interm, on=merge_keys, suffixes=('', '_nn'))
    return merged


def get_feature_columns(df):
    # separo quines columnes son features manuals i quines son les 256 dim de interm
    manual_cols = [c for c in df.columns
                   if c not in NON_FEATURE_COLS and not c.startswith('interm_')]
    interm_cols = [c for c in df.columns if c.startswith('interm_')]
    return manual_cols, interm_cols


def evaluate_feature(df, manual_feature, interm_cols, level):
    """
    Entrena un Random Forest per predir 'manual_feature' a partir de les
    256 dimensions interm. Retorna R² mitjà i desviació via cross-validation.
    """
    y = df[manual_feature].astype(float)
    valid = y.notna()

    if valid.sum() < N_SPLITS * 2:   # cal prou mostres per fer CV
        return None

    X = df.loc[valid, interm_cols].values
    y = y[valid].values

    if level == "per_slice":
        # GroupKFold: les slices del mateix pacient mai es separen entre folds
        groups = df.loc[valid, 'PatientID'].values
        n_groups = len(np.unique(groups))
        if n_groups < N_SPLITS:
            return None
        cv = GroupKFold(n_splits=N_SPLITS)
        splitter = cv.split(X, y, groups=groups)
    else:
        # per pacient ja no cal groupkfold, cada fila es un pacient diferent
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        splitter = cv.split(X, y)

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                   random_state=RANDOM_STATE, n_jobs=-1)

    scores = []
    for train_idx, test_idx in splitter:
        model.fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[test_idx], y[test_idx]))

    return {
        'R2_mean': np.mean(scores),
        'R2_std':  np.std(scores),
        'N':       valid.sum()
    }


def plot_results(results_df, title, output_png):
    # nomes ensenyo les 25 features amb millor R2, amb colors segons si es bo/moderat/dolent
    top = results_df.sort_values('R2_mean', ascending=False).head(25)
    colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red'
              for v in top['R2_mean']]

    plt.figure(figsize=(12, max(6, len(top) * 0.35)))
    plt.barh(top['Manual_Feature'], top['R2_mean'], color=colors, alpha=0.75,
            xerr=top['R2_std'], error_kw={'capsize': 3, 'color': 'black'})
    plt.axvline(x=0.7, color='green',  linestyle='--', alpha=0.5, label='R²=0.7 (bo)')
    plt.axvline(x=0.4, color='orange', linestyle='--', alpha=0.5, label='R²=0.4 (moderat)')
    plt.axvline(x=0,   color='black',  linewidth=0.8)
    plt.xlabel('R² (cross-validation)')
    plt.title(title, fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()


def process_combination(dataset, phase, level):
    # per cada combinacio (dataset, phase, level) carrego, avaluo totes les features i guardo resultats
    prefix = f"{dataset}_{phase}_{level}"
    print(f"\n{'='*60}")
    print(f"Processant: {prefix}")
    print(f"{'='*60}")

    merged = load_pair(dataset, phase, level)
    if merged is None:
        return

    manual_cols, interm_cols = get_feature_columns(merged)
    print(f"  Features manuals: {len(manual_cols)}")
    print(f"  Dimensions interm: {len(interm_cols)}")
    print(f"  Files totals:      {len(merged)}")
    if level == "per_slice":
        print(f"  Pacients únics:    {merged['PatientID'].nunique()}")

    results = []
    for mc in manual_cols:
        res = evaluate_feature(merged, mc, interm_cols, level)
        if res is None:
            print(f"    {mc:<25} → SKIP (dades insuficients)")
            continue
        results.append({'Manual_Feature': mc, **res})
        print(f"    {mc:<25} R²={res['R2_mean']:.4f} ± {res['R2_std']:.4f}  (N={res['N']})")

    if not results:
        print("  Cap feature avaluable.")
        return

    results_df = pd.DataFrame(results).sort_values('R2_mean', ascending=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_out = os.path.join(OUTPUT_DIR, f"{prefix}_multivariate_R2.csv")
    results_df.to_csv(csv_out, index=False)
    print(f"\n  Guardat: {csv_out}")

    png_out = os.path.join(OUTPUT_DIR, f"{prefix}_multivariate_R2.png")
    plot_results(results_df,
                title=f"Random Forest: feature manual ← 256 dim. interm — "
                      f"{dataset.upper()} {phase} ({level})",
                output_png=png_out)
    print(f"  Guardat: {png_out}")


def main():
    # recorro totes les combinacions de dataset x phase x level i les processo una a una
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for dataset in DATASETS:
        for phase in PHASES:
            for level in LEVELS:
                process_combination(dataset, phase, level)

    print(f"\n{'='*60}")
    print(f"Fet! Resultats guardats a: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
