# config.py
import os
import random
import numpy as np
import torch
from pathlib import Path

# ================================
# PARÀMETRES GLOBALS
# ================================
SEED = 42
Phase = "ES"

# LR baix: essencial per fine-tuning de ResNet pre-entrenada
LEARNING_RATE = 1e-4

VAR_THRESHOLD = 0.02
FILTER_EMPTY_SLICES = True
BATCH_SIZE = 16
NUM_EPOCHS =300

# ResNet18 espera 224x224 (era 128x128 → incorrecte)
TARGET_SIZE = (224, 224)

# early stopping desactivat de moment (queda implementat per si el vull activar despres)
USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 15

# Normalització ImageNet — OBLIGATORI per ResNet pre-entrenada
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Directoris
BASE_DIR_TRAIN = Path(r"C:\Users\Alex\Downloads\Projecte_estiu\ACDC\ACDC\database\training")
BASE_DIR_TEST  = Path(r"C:\Users\Alex\Downloads\Projecte_estiu\ACDC\ACDC\database\testing")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
