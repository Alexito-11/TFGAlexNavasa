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

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def create_cnn_model(n_classes, device):
    """Crea la CNN des de zero (pesos inicialitzats aleatòriament)."""
    model = SimpleCNN(n_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Model CNN creat:")
    print(f"   Classes: {n_classes}")
    print(f"   Paràmetres entrenables: {n_params:,}")
    return model