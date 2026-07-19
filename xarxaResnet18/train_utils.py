# train_utils.py
# coses compartides pels trainers: early stopping, optimizer (amb les 2 fases
# de la resnet), scheduler i la loss

import torch.optim as optim
import torch.nn as nn
import config


class EarlyStopping:
    """Early stopping per evitar overfitting"""
    # atura l'entrenament si la val_loss no millora durant "patience" epoques seguides

    def __init__(self, patience=15, min_delta=1e-4):
        self.patience  = patience    # epoques sense millora abans d'aturar
        self.min_delta = min_delta   # marge minim per considerar que hi ha millora
        self.counter   = 0           # comptador d'epoques sense millora
        self.best_loss = None        # millor val_loss vista fins ara

    def __call__(self, val_loss):
        if self.best_loss is None:
            # primera epoca, guardo com a referencia
            self.best_loss = val_loss
            return False
        if val_loss > self.best_loss - self.min_delta:
            # no ha millorat prou
            self.counter += 1
            if self.counter >= self.patience:
                return True  # ja no aguanto mes, aturo
        else:
            # ha millorat de veritat, reinicio comptador
            self.counter   = 0
            self.best_loss = val_loss
        return False


def get_optimizer(model, unfreeze_backbone=False):
    """
    Fase 1 (backbone congelat):
        Entrena només intermediate + classifier amb lr=1e-4.

    Fase 2 (fine-tuning, backbone descongelat):
        Backbone amb lr 10x menor (1e-5).
        intermediate + classifier amb lr normal (1e-4).
    """
    # Cap nou = intermediate + classifier
    head_params = list(model.intermediate.parameters()) + \
                  list(model.classifier.parameters())

    if unfreeze_backbone:
        # fase 2: backbone descongelat, li poso lr mes baix perque ja ve preentrenat
        backbone_layers = [model.conv1, model.bn1,
                           model.layer1, model.layer2,
                           model.layer3, model.layer4]
        backbone_params = []
        for layer in backbone_layers:
            backbone_params += list(layer.parameters())

        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': config.LEARNING_RATE * 0.1},
            {'params': head_params,     'lr': config.LEARNING_RATE}
        ], weight_decay=1e-3)
        print("  Optimizer: fine-tuning (backbone lr=1e-5 | cap lr=1e-4)")
    else:
        # fase 1: backbone congelat, nomes entreno el cap
        optimizer = optim.Adam(head_params,
                               lr=config.LEARNING_RATE,
                               weight_decay=1e-3)
        print("  Optimizer: només cap (lr=1e-4)")

    return optimizer


def get_scheduler(optimizer):
    # baixa el lr a la meitat si la val_loss no millora en 8 epoques
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )


def get_criterion():
    # cross entropy pq es classificacio multiclasse
    return nn.CrossEntropyLoss()
