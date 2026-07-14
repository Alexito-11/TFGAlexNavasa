# data_analysis.py
import numpy as np
import matplotlib.pyplot as plt


def analyze_global_images(X_all):
    """Anàlisi global de totes les imatges"""
    print("\n" + "=" * 60)
    print("ANALISI GLOBAL DE LES IMATGES")
    print("=" * 60)

    print(f"Total d'imatges analitzades: {X_all.shape[0]}")
    print(f"Mida de cada imatge: {X_all.shape[1]}x{X_all.shape[2]}")
    print(f"Valor minim global: {X_all.min():.3f}")
    print(f"Valor maxim global: {X_all.max():.3f}")
    print(f"Mitjana global: {X_all.mean():.3f}")
    print(f"Desviacio estandard global: {X_all.std():.3f}")

    return X_all


def plot_sample_images(X_all, num_exemples=5):
    """Mostra exemples d'imatges"""
    idx_mostrar = np.random.choice(len(X_all), min(num_exemples, len(X_all)), replace=False)

    fig, axes = plt.subplots(1, len(idx_mostrar), figsize=(15, 3))
    if len(idx_mostrar) == 1:
        axes = [axes]

    for i, idx in enumerate(idx_mostrar):
        axes[i].imshow(X_all[idx], cmap='gray')
        axes[i].set_title(f'Imatge {idx}')
        axes[i].axis('off')

    plt.suptitle('Exemples d\'imatges del dataset')
    plt.tight_layout()
    plt.show()


def plot_global_histogram(X_all):
    """Histograma global de tots els píxels"""
    plt.figure(figsize=(10, 5))
    plt.hist(X_all.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    plt.axvline(X_all.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mitjana: {X_all.mean():.2f}')
    plt.axvline(X_all.mean() + X_all.std(), color='orange', linestyle='--', linewidth=2,
                label=f'+1σ: {X_all.mean() + X_all.std():.2f}')
    plt.axvline(X_all.mean() - X_all.std(), color='orange', linestyle='--', linewidth=2,
                label=f'-1σ: {X_all.mean() - X_all.std():.2f}')
    plt.xlabel('Valor del pixel')
    plt.ylabel('Frequencia')
    plt.title('Histograma global de tots els pixels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def analyze_background(X_all):
    """Analitza el background de les imatges"""
    threshold = 0.1 * X_all.max()
    pixels_bg = X_all[X_all <= threshold]
    pixels_fg = X_all[X_all > threshold]

    print(f"\nANALISI DE BACKGROUND:")
    print(f"  Threshold (10% del maxim): {threshold:.3f}")
    print(f"  Pixels de background: {len(pixels_bg)} ({len(pixels_bg) / X_all.size * 100:.1f}%)")
    print(f"  Pixels de foreground: {len(pixels_fg)} ({len(pixels_fg) / X_all.size * 100:.1f}%)")
    print(f"  Mitjana del foreground: {pixels_fg.mean():.3f}")
    print(f"  Desviacio del foreground: {pixels_fg.std():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(X_all.flatten(), bins=100, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Valor del pixel')
    axes[0].set_ylabel('Frequencia')
    axes[0].set_title('Tots els pixels')
    axes[0].axvline(X_all.mean(), color='red', linestyle='--', label=f'Mitjana: {X_all.mean():.2f}')
    axes[0].axvline(threshold, color='purple', linestyle=':', label=f'Threshold: {threshold:.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(pixels_fg, bins=50, color='green', alpha=0.7)
    axes[1].set_xlabel('Valor del pixel')
    axes[1].set_ylabel('Frequencia')
    axes[1].set_title('Nomes foreground (sense background)')
    axes[1].axvline(pixels_fg.mean(), color='red', linestyle='--', label=f'Mitjana: {pixels_fg.mean():.2f}')
    axes[1].axvline(pixels_fg.mean() + pixels_fg.std(), color='orange', linestyle='--',
                    label=f'+1σ: {pixels_fg.mean() + pixels_fg.std():.2f}')
    axes[1].axvline(pixels_fg.mean() - pixels_fg.std(), color='orange', linestyle='--',
                    label=f'-1σ: {pixels_fg.mean() - pixels_fg.std():.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Comparacio d\'histogrames')
    plt.tight_layout()
    plt.show()


def analyze_class_balance(y_train_enc, y_val_enc, y_test_enc, label_encoder):
    """Analitza el balanceig de classes"""
    print("\n" + "=" * 60)
    print("ANALISI DE BALANCEIG PER SLICES")
    print("=" * 60)

    train_counts = np.bincount(y_train_enc)
    val_counts   = np.bincount(y_val_enc)
    test_counts  = np.bincount(y_test_enc)

    print(f"\nTRAIN ({len(y_train_enc)} slices):")
    for i, count in enumerate(train_counts):
        if count > 0:
            class_name = label_encoder.inverse_transform([i])[0]
            percentage = 100 * count / len(y_train_enc)
            print(f"  {class_name}: {count} slices ({percentage:.1f}%)")

    print(f"\nVALIDATION ({len(y_val_enc)} slices):")
    for i, count in enumerate(val_counts):
        if count > 0:
            class_name = label_encoder.inverse_transform([i])[0]
            percentage = 100 * count / len(y_val_enc)
            print(f"  {class_name}: {count} slices ({percentage:.1f}%)")

    print(f"\nTEST ({len(y_test_enc)} slices):")
    for i, count in enumerate(test_counts):
        if count > 0:
            class_name = label_encoder.inverse_transform([i])[0]
            percentage = 100 * count / len(y_test_enc)
            print(f"  {class_name}: {count} slices ({percentage:.1f}%)")

    print("\n" + "=" * 60)
    print("METRIQUES DE DESBALEIG")
    print("=" * 60)

    train_ratio = max(train_counts) / min(train_counts[train_counts > 0])
    val_ratio   = max(val_counts)   / min(val_counts[val_counts > 0])   if len(val_counts)  > 0 else 0
    test_ratio  = max(test_counts)  / min(test_counts[test_counts > 0]) if len(test_counts) > 0 else 0

    print(f"Coeficient de desbalanceig (majoritaria/minoritaria):")
    print(f"  TRAIN: {train_ratio:.2f} : 1")
    print(f"  VAL:   {val_ratio:.2f} : 1")
    print(f"  TEST:  {test_ratio:.2f} : 1")

    return train_counts, val_counts, test_counts


def plot_class_distribution(train_counts, val_counts, test_counts, label_encoder,
                            y_train_enc, y_val_enc, y_test_enc):
    """Gràfics de distribució de classes"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    bars1 = ax1.bar(label_encoder.classes_, train_counts, color='blue', alpha=0.7)
    ax1.set_xlabel('Classe')
    ax1.set_ylabel('Nombre de slices')
    ax1.set_title(f'Distribucio per classe - TRAIN (total: {len(y_train_enc)} slices)')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars1, train_counts):
        percentage = 100 * count / len(y_train_enc)
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)

    ax2 = axes[0, 1]
    bars2 = ax2.bar(label_encoder.classes_, val_counts, color='orange', alpha=0.7)
    ax2.set_xlabel('Classe')
    ax2.set_ylabel('Nombre de slices')
    ax2.set_title(f'Distribucio per classe - VALIDATION (total: {len(y_val_enc)} slices)')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars2, val_counts):
        percentage = 100 * count / len(y_val_enc)
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)

    ax3 = axes[1, 0]
    bars3 = ax3.bar(label_encoder.classes_, test_counts, color='red', alpha=0.7)
    ax3.set_xlabel('Classe')
    ax3.set_ylabel('Nombre de slices')
    ax3.set_title(f'Distribucio per classe - TEST (total: {len(y_test_enc)} slices)')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars3, test_counts):
        percentage = 100 * count / len(y_test_enc)
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()