# mainxarxa0.py
# Script principal per entrenar i avaluar la CNN des de zero (SimpleCNN)
# sobre el dataset ACDC, seguint el mateix pipeline de dades i el mateix
# pressupost d'entrenament (LR i èpoques) que la ResNet18, per poder-les comparar.

import os
import shutil
import numpy as np
import torch

import config
from data_loader import (
    load_dataset,
    patient_stratified_split,
    build_slice_arrays,
    apply_oversampling,
    create_dataloaders,
)
from modelxarxa0 import create_cnn_model
from trainerxarxa0 import TrainerCNN
from evaluationxarxa0 import predict_patients, analyze_patients


# Igualtat de condicions amb la ResNet: mateix LR base i mateix pressupost d'èpoques
LR = config.LEARNING_RATE           # 1e-4
NUM_EPOCHS_CNN = config.NUM_EPOCHS   # 300
CHECKPOINT_PATH = "best_model_cnn.pth"
CURVES_PNG = "cnn_training_curves.png"


def main():
    # Fixa la llavor aleatòria per assegurar reproduïbilitat i selecciona GPU si n'hi ha
    config.set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositiu: {device}")

    # 1. Carregar dades TRAIN
    print("\nCarregant dades d'entrenament...")
    train_patient_data, train_patient_to_group = load_dataset(
        config.BASE_DIR_TRAIN, config.Phase
    )

    # 2. Split estratificat train/val per pacient
    #    (garanteix que totes les slices d'un mateix pacient van al mateix conjunt,
    #    evitant fuita de dades entre train i validació)
    train_ids, val_ids, le = patient_stratified_split(
        train_patient_data, train_patient_to_group, test_size=0.2
    )

    # 3. Construir arrays de slices a partir dels IDs de pacient de cada split
    X_train, y_train = build_slice_arrays(
        train_ids, train_patient_data, train_patient_to_group
    )
    X_val, y_val = build_slice_arrays(
        val_ids, train_patient_data, train_patient_to_group
    )

    # 4. Oversampling només a train, per compensar el desequilibri de classes
    #    sense contaminar el conjunt de validació
    X_train, y_train = apply_oversampling(X_train, y_train)

    # 5. Codificar labels (text → enters) reutilitzant el mateix LabelEncoder (le)
    #    per a train, val i test, per garantir la mateixa correspondència de classes
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    class_names = list(le.classes_)

    # 6. Carregar dades TEST (conjunt independent, mai vist durant l'entrenament)
    print("\nCarregant dades de test...")
    test_patient_data, test_patient_to_group = load_dataset(
        config.BASE_DIR_TEST, config.Phase
    )

    test_ids = np.array(list(test_patient_data.keys()))
    X_test, y_test = build_slice_arrays(
        test_ids, test_patient_data, test_patient_to_group
    )
    y_test_enc = le.transform(y_test)

    # 7. Crear dataloaders per als tres conjunts (train, val, test)
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc
    )

    # 8. Crear model CNN amb pesos inicialitzats aleatòriament (des de zero)
    print("\nCreant model CNN des de zero...")
    n_classes = len(class_names)
    model = create_cnn_model(n_classes=n_classes, device=device)

    # 9. Entrenar la CNN amb el mateix lr i weight_decay que la ResNet,
    #    per fer la comparació entre models el més justa possible
    trainer = TrainerCNN(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_names=class_names,
        lr=LR,
        weight_decay=1e-3,          # igual que la ResNet
    )
    history = trainer.train(NUM_EPOCHS_CNN, checkpoint_path=CHECKPOINT_PATH)

    # 10. Renombrar el fitxer de corbes genèric a un nom específic de la CNN,
    #     per no sobreescriure les corbes de la ResNet en executar tots dos scripts
    if os.path.exists("training_curves.png"):
        shutil.move("training_curves.png", CURVES_PNG)
        print(f"Corbes renombrades a: {CURVES_PNG}")

    # 11. Carregar el millor model desat durant l'entrenament (segons checkpoint)
    #     per fer l'avaluació final amb els pesos òptims, no amb els de l'última època
    print("\nCarregant millor model per avaluar...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    print(f"Checkpoint carregat: època {ckpt['epoch']}, val_acc {ckpt['val_acc']:.4f}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 12. Reconstruir diccionaris de train i val per pacient a partir dels IDs
    #     obtinguts al split del pas 2, necessaris per a l'avaluació per pacient
    train_pdata  = {pid: train_patient_data[pid] for pid in train_ids}
    train_pgroup = {pid: train_patient_to_group[pid] for pid in train_ids}
    val_pdata    = {pid: train_patient_data[pid] for pid in val_ids}
    val_pgroup   = {pid: train_patient_to_group[pid] for pid in val_ids}

    # 13. Avaluació per pacient als tres conjunts (train, validació, test):
    #     is_resnet=False indica al mòdul d'avaluació que utilitzi el pipeline
    #     de preprocessat/predicció corresponent a la CNN
    for split_name, pdata, pgroup in [
        ("TRAIN", train_pdata, train_pgroup),
        ("VALIDACIÓ", val_pdata, val_pgroup),
        ("TEST", test_patient_data, test_patient_to_group),
    ]:
        print(f"\n{'#'*60}")
        print(f"# AVALUACIÓ PER PACIENT — {split_name}")
        print(f"{'#'*60}")
        results = predict_patients(
            model, pdata, pgroup, le, device, is_resnet=False
        )
        analyze_patients(results, le, dataset_name=split_name)


if __name__ == "__main__":
    main()
