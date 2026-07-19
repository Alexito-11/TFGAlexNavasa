# univariate_correlation.py
# Anàlisi de correlació univariada (Pearson) entre cada feature manual
# i cada dimensió 'interm' (256) de la xarxa neuronal.
#
# Es processen automàticament totes les combinacions:
#   dataset  ∈ {training, testing}
#   phase    ∈ {ED, ES}
#   level    ∈ {per_slice, per_pacient}
#
# Per cada combinació es genera:
#   - {prefix}_correlation_matrix.csv   → matriu completa (n_manual x 256)
#   - {prefix}_top_correlations.csv     → millor interm per cada feature manual
#   - {prefix}_heatmap.png              → visualització amb llindar i clustering

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# CONFIGURACIÓ
INPUT_DIR  = r"C:\Users\Alex\PycharmProjects\PythonProject"
OUTPUT_DIR = r"C:\Users\Alex\PycharmProjects\PythonProject\correlacions_univariades"

DATASETS = ["training", "testing"]
PHASES   = ["ED", "ES"]
LEVELS   = ["per_slice", "per_pacient"]   # nivells d'agregació

THRESHOLD = 0.6   # llindar per a la visualització (no afecta el càlcul ni el CSV complet)
TOP_N     = 5     # nombre de millors dimensions interm a guardar per cada feature manual

# columnes que no son features (metadata del pacient), per no ficar-les al calcul
NON_FEATURE_COLS = {
    'PatientID', 'Phase', 'Slice_No', 'Label',
    'Patient_group', 'Height', 'Weight', 'Patient_Information'
}


def load_pair(dataset, phase, level):
    """Carrega el CSV manual i el CSV interm corresponents, i els ajunta (merge)."""
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

    # Claus de merge segons el nivell
    merge_keys = ['PatientID', 'Phase']
    if level == "per_slice":
        merge_keys.append('Slice_No')

    merged = pd.merge(df_manual, df_interm, on=merge_keys, suffixes=('', '_nn'))
    return merged


def get_feature_columns(df):
    # separo columnes manuals de les 256 dim interm de la xarxa
    manual_cols = [c for c in df.columns
                   if c not in NON_FEATURE_COLS and not c.startswith('interm_')]
    interm_cols = [c for c in df.columns if c.startswith('interm_')]
    return manual_cols, interm_cols


def compute_correlation_matrix(df, manual_cols, interm_cols):
    """
    Calcula la matriu de correlació de Pearson (n_manual x n_interm).
    Usa pairwise-complete (ignora NaN automàticament per cada parella de columnes).
    """
    n_manual = len(manual_cols)
    n_interm = len(interm_cols)
    corr_matrix = np.full((n_manual, n_interm), np.nan)

    for i, mc in enumerate(manual_cols):
        x = df[mc].astype(float)
        for j, ic in enumerate(interm_cols):
            y = df[ic].astype(float)
            valid = x.notna() & y.notna()
            if valid.sum() < 3:           # cal almenys 3 punts per correlar
                continue
            xv, yv = x[valid], y[valid]
            if xv.std() == 0 or yv.std() == 0:   # constant → correlació indefinida
                continue
            corr_matrix[i, j] = np.corrcoef(xv, yv)[0, 1]

    return pd.DataFrame(corr_matrix, index=manual_cols, columns=interm_cols)


def save_top_correlations(corr_df, output_csv, top_n=TOP_N):
    """
    Per cada feature manual, guarda les top_n dimensions interm amb major |correlació|.
    Format long: una fila per (feature, rang).
    """
    results = []
    for mc in corr_df.index:
        row = corr_df.loc[mc].dropna()
        if row.empty:
            continue
        # Ordenar per valor absolut descendent i agafar les top_n
        top = row.reindex(row.abs().sort_values(ascending=False).index).head(top_n)
        for rank, (interm_name, val) in enumerate(top.items(), start=1):
            results.append({
                'Manual_Feature': mc,
                'Rank':           rank,
                'Interm':         interm_name,
                'Correlation':    round(val, 4)
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Ordenar globalment per feature i rang
        results_df = results_df.sort_values(['Manual_Feature', 'Rank'])
    results_df.to_csv(output_csv, index=False)
    return results_df


def _cluster_order(matrix, axis=0):
    """
    Retorna l'ordre dels índexs de `matrix` segons clustering jeràrquic
    (Ward) sobre la distància 1 - |correlació| entre files (axis=0) o
    columnes (axis=1). Retorna np.arange(...) si no és possible.
    """
    if axis == 1:
        matrix = matrix.T
    n = matrix.shape[0]
    if n < 3:
        return np.arange(n)
    # Correlació entre files
    corr = np.corrcoef(matrix)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1 - np.abs(corr)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    try:
        link  = linkage(squareform(dist, checks=False), method='ward')
        order = leaves_list(link)
    except Exception:
        # si el clustering falla per algun motiu, deixo l'ordre original
        order = np.arange(n)
    return order


def plot_heatmap(corr_df, threshold, title, output_png):
    """Heatmap amb llindar i clustering jeràrquic tant a files com a columnes."""
    corr_clean = corr_df.fillna(0).values
    manual_cols = list(corr_df.index)
    interm_cols = list(corr_df.columns)

    # Aplicar llindar
    corr_thresh = corr_clean.copy()
    corr_thresh[np.abs(corr_thresh) < threshold] = 0

    # Clustering a files (features manuals) sobre matriu original amb signe
    row_order = _cluster_order(corr_clean, axis=0)

    # Clustering a columnes (dimensions interm) sobre matriu original amb signe
    col_order = _cluster_order(corr_clean, axis=1)

    # Reordenar segons ambdós ordres
    corr_ordered   = corr_thresh[np.ix_(row_order, col_order)]
    manual_ordered = [manual_cols[i] for i in row_order]
    interm_ordered = [interm_cols[j] for j in col_order]

    fig, ax = plt.subplots(figsize=(28, max(6, len(manual_ordered) * 0.3)))
    im = ax.imshow(corr_ordered, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)

    # nomes mostro 1 de cada "step" etiquetes a l'eix x perque no se solapin (son 256)
    step = max(1, len(interm_ordered) // 30)
    ax.set_xticks(range(0, len(interm_ordered), step))
    ax.set_xticklabels(interm_ordered[::step], rotation=90, fontsize=6)
    ax.set_xlabel(f'Dimensions interm ({len(interm_ordered)}, agrupades per similitud)',
                  fontsize=10)

    ax.set_yticks(range(len(manual_ordered)))
    ax.set_yticklabels(manual_ordered, fontsize=7)
    ax.set_ylabel('Features manuals (agrupades per similitud)', fontsize=10)

    ax.set_title(f'{title}\n(llindar |r|={threshold}; valors amb |r|<llindar visualitzats com 0)',
                 fontsize=11)
    plt.colorbar(im, ax=ax, label='Correlació de Pearson', shrink=0.6)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()


def process_combination(dataset, phase, level):
    # per cada combinacio calculo la matriu de correlacio i genero els 3 outputs
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
    print(f"  Files (mostres):   {len(merged)}")

    corr_df = compute_correlation_matrix(merged, manual_cols, interm_cols)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # matriu completa
    matrix_csv = os.path.join(OUTPUT_DIR, f"{prefix}_correlation_matrix.csv")
    corr_df.to_csv(matrix_csv)
    print(f"  Guardat: {matrix_csv}")

    # top correlacions per feature
    top_csv = os.path.join(OUTPUT_DIR, f"{prefix}_top_correlations.csv")
    top_df  = save_top_correlations(corr_df, top_csv)
    print(f"  Guardat: {top_csv}")
    if not top_df.empty:
        # Mostra les 3 correlacions absolutes més altes globalment
        top_global = top_df.reindex(top_df['Correlation'].abs().sort_values(ascending=False).index)
        print(f"  Top 3 correlacions globals:")
        for _, r in top_global.head(3).iterrows():
            print(f"    {r['Manual_Feature']:<25} ↔ {r['Interm']:<12} r={r['Correlation']:.4f}")

    # heatmap amb clustering
    heatmap_png = os.path.join(OUTPUT_DIR, f"{prefix}_heatmap.png")
    plot_heatmap(corr_df, THRESHOLD,
                title=f"Correlació univariada — {dataset.upper()} {phase} ({level})",
                output_png=heatmap_png)
    print(f"  Guardat: {heatmap_png}")


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
