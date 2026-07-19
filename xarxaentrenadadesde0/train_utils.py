# train_utils.py
# Mòdul d'utilitats compartides per l'entrenament de les xarxes (ResNet i CNN des de zero).
# Centralitza tres peces reutilitzades pels diferents Trainers: l'early stopping,
# la creació de l'optimitzador (amb suport per a les fases de fine-tuning de la ResNet),
# el scheduler de learning rate i la funció de pèrdua.

import torch.optim as optim
import torch.nn as nn
import config


class EarlyStopping:
    """Early stopping per evitar overfitting"""
    # Atura l'entrenament si la val_loss no millora durant un nombre determinat
    # d'èpoques consecutives (patience), evitant que el model es sobreajusti
    # continuant a entrenar-se un cop ja ha deixat de generalitzar bé.

    def __init__(self, patience=15, min_delta=1e-4):
        self.patience  = patience    # nombre d'èpoques sense millora abans d'aturar
        self.min_delta = min_delta   # marge mínim de millora per considerar-la real
        self.counter   = 0           # comptador d'èpoques consecutives sense millora
        self.best_loss = None        # millor val_loss vista fins ara

    def __call__(self, val_loss):
        # Es crida a cada època passant-li la val_loss actual;
        # retorna True quan cal aturar l'entrenament (s'ha exhaurit la paciència)
        if self.best_loss is None:
            # Primera època: es fixa com a referència inicial
            self.best_loss = val_loss
            return False
        if val_loss > self.best_loss - self.min_delta:
            # La pèrdua no ha millorat prou respecte al mínim anterior
            self.counter += 1
            if self.counter >= self.patience:
                return True  # s'ha superat la paciència: cal aturar
        else:
            # Hi ha una millora real: es reinicia el comptador i s'actualitza el mínim
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
    # Construeix l'optimitzador Adam adaptat a la fase d'entrenament de la ResNet.
    # Cap nou = intermediate + classifier (les capes afegides sobre la ResNet preentrenada,
    # que sempre s'entrenen amb el learning rate normal)
    head_params = list(model.intermediate.parameters()) + \
                  list(model.classifier.parameters())

    if unfreeze_backbone:
        # Fase 2: fine-tuning. Es descongela tot el backbone de la ResNet
        # (les capes convolucionals originals preentrenades sobre ImageNet)
        backbone_layers = [model.conv1, model.bn1,
                           model.layer1, model.layer2,
                           model.layer3, model.layer4]
        backbone_params = []
        for layer in backbone_layers:
            backbone_params += list(layer.parameters())

        # S'utilitzen dos grups de paràmetres amb learning rates diferents:
        # el backbone (ja preentrenat) amb un lr molt més baix per no destruir
        # el coneixement apres, i el cap nou amb el lr normal
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': config.LEARNING_RATE * 0.1},
            {'params': head_params,     'lr': config.LEARNING_RATE}
        ], weight_decay=1e-3)
        print("  Optimizer: fine-tuning (backbone lr=1e-5 | cap lr=1e-4)")
    else:
        # Fase 1: backbone congelat. Només s'entrenen les capes noves (cap),
        # ja que el backbone es manté fix amb els pesos d'ImageNet
        optimizer = optim.Adam(head_params,
                               lr=config.LEARNING_RATE,
                               weight_decay=1e-3)
        print("  Optimizer: només cap (lr=1e-4)")

    return optimizer


def get_scheduler(optimizer):
    # Scheduler compartit per tots els Trainers: redueix el learning rate a la meitat
    # (factor=0.5) si la mètrica monitoritzada (val_loss, mode='min') no millora
    # durant 8 èpoques consecutives (patience=8)
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )


def get_criterion():
    # Funció de pèrdua compartida per tots els models: entropia creuada,
    # estàndard per a problemes de classificació multiclasse
    return nn.CrossEntropyLoss()
