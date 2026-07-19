# modelxarxa0.py
# CNN entrenada des de zero per classificar les 5 patologies cardíaques del dataset ACDC.
# Arquitectura senzilla (però amb prou capacitat) per servir com a comparació amb la
# ResNet18 preentrenada.

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    CNN amb 4 blocs conv (32→64→128→256 canals), BatchNorm, MaxPool i FC.

    Respecte a la versió anterior (16→32→64→128), es dobla el nombre de canals a
    cada bloc. Això dona una mica més de capacitat per mirar de reduir el
    subajust observat (accuracy d'entrenament baixa), mantenint-la, tot i així,
    molt més simple que la ResNet18 (11M de paràmetres).
    """

    def __init__(self, n_classes=5, dropout=0.4):
        super().__init__()

        # Bloc extractor de característiques: 4 blocs conv+BN+ReLU+MaxPool
        # que van augmentant el nombre de canals (32→64→128→256) mentre
        # redueixen progressivament la resolució espacial (assumint entrada 128x128)
        self.features = nn.Sequential(
            # Block 1: 128 → 64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 64 → 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 32 → 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 16 → 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Global Average Pooling: redueix el mapa 8x8x256 a 1x1x256,
        # fent el model invariant a la mida exacta de l'entrada
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Capçalera classificadora: aplana el vector 256, passa per una capa
        # oculta de 128 neurones amb Dropout (regularització) i acaba amb
        # la capa de sortida amb una neurona per classe
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # Passa seqüencial: extracció de característiques → pooling global → classificació
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def create_cnn_model(n_classes, device):
    """Crea la CNN des de zero (pesos inicialitzats aleatòriament)."""
    # Instancia el model i el mou al dispositiu (CPU/GPU) indicat
    model = SimpleCNN(n_classes=n_classes).to(device)
    # Compta només els paràmetres entrenables (requires_grad=True),
    # útil per comparar la capacitat d'aquest model amb la de la ResNet18
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Model CNN creat:")
    print(f"   Classes: {n_classes}")
    print(f"   Paràmetres entrenables: {n_params:,}")
    return model
