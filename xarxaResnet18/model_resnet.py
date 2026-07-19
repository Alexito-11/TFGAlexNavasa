# model_resnet.py
# resnet18 preentrenada amb imagenet, li trec la capa final i li afegeixo
# una capa intermedia + classificador propi per les 5 classes de l'ACDC

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class FeatureExtractor(nn.Module):
    """ResNet18 amb extracció de característiques a múltiples nivells"""

    def __init__(self, base_model, intermediate_size=256, n_classes=5):
        super().__init__()
        # agafo les capes del backbone de la resnet ja preentrenada, una a una
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = base_model.avgpool

        # capa nova entre el backbone i el classificador, redueix a intermediate_size
        self.intermediate = nn.Sequential(
            nn.Linear(base_model.fc.in_features, intermediate_size),
            nn.ReLU()
        )

        # Classificador final amb n_classes (no només 2!)
        self.classifier = nn.Linear(intermediate_size, n_classes)

        # per guardar les features intermedies si les vull mirar despres
        self.features_all = []

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features1 = self.maxpool(x)

        features2 = self.layer1(features1)
        features3 = self.layer2(features2)
        features4 = self.layer3(features3)
        features5 = self.layer4(features4)

        pooled = self.avgpool(features5)
        pooled = torch.flatten(pooled, 1)

        interm = self.intermediate(pooled)

        # guardo pooled i interm per si vull analitzar-les fora (feature importance, etc)
        self.features_all = [pooled, interm]

        out = self.classifier(interm)
        return out


def create_resnet_model(n_classes, device, intermediate_size=256):
    """Crea el model ResNet18 preentrenat amb FeatureExtractor"""

    # carrego la resnet18 amb els pesos d'imagenet
    base_resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = FeatureExtractor(base_resnet, intermediate_size=intermediate_size, n_classes=n_classes)
    model = model.to(device)

    print(f" Model ResNet18 FeatureExtractor creat:")
    print(f"   Classes: {n_classes}")
    print(f"   Intermediate size: {intermediate_size}")
    print(f"   Pesos ImageNet carregats")

    return model


def load_pretrained_resnet(checkpoint_path, n_classes, device, intermediate_size=256):
    """Carrega un model ResNet18 FeatureExtractor preentrenat"""

    # aqui no cal carregar pesos d'imagenet, ja els sobreescriurem amb el checkpoint
    base_resnet = models.resnet18(weights=None)
    model = FeatureExtractor(base_resnet, intermediate_size=intermediate_size, n_classes=n_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n Model ResNet18 carregat:")
        print(f"   Època: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Val Acc: {checkpoint.get('val_acc', checkpoint.get('best_val_acc', 'N/A'))}")
    else:
        # checkpoint "pelat", nomes els pesos sense metadata
        model.load_state_dict(checkpoint)
        print(f"\n Pesos del model ResNet18 carregats correctament")

    model = model.to(device)
    model.eval()

    return model
