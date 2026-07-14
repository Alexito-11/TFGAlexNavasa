# main_resnet.py
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

import config
from data_loader import (
    load_dataset, patient_stratified_split, build_slice_arrays,
    apply_oversampling, create_dataloaders,
    SliceDataset, get_val_transforms
)
from data_analysis import (
    analyze_global_images, plot_sample_images, plot_global_histogram,
    analyze_background, analyze_class_balance, plot_class_distribution
)
from model_resnet import create_resnet_model, load_pretrained_resnet
from trainer import Trainer
from evaluation import predict_model, evaluate_slices, predict_patients, analyze_patients


def main():
    config.set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositiu: {device}")

    # ──────────────────────────────────────────────
    # CONFIGURACIÓ
    # ──────────────────────────────────────────────
    USE_PRETRAINED        = False
    PRETRAINED_MODEL_PATH = "best_model_resnet18.pth"
    INTERMEDIATE_SIZE     = 256
    EPOCHS_PHASE1         = 30

    print("\n" + "="*60)
    print("CONFIGURACIÓ RESNET18")
    print("="*60)
    print(f"  Fase ACDC:       {config.Phase}")
    print(f"  Target size:     {config.TARGET_SIZE}")
    print(f"  Learning rate:   {config.LEARNING_RATE}")
    print(f"  Batch size:      {config.BATCH_SIZE}")
    print(f"  Èpoques totals:  {config.NUM_EPOCHS}")
    print(f"  Èpoques fase 1:  {EPOCHS_PHASE1}")
    print(f"  Early stopping:  {config.USE_EARLY_STOPPING}")
    print(f"  Use pretrained:  {USE_PRETRAINED}")

    # ──────────────────────────────────────────────
    # 1. CÀRREGA DE DADES
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("CÀRREGA DE DADES")
    print("="*60)

    patient_data_train, patient_to_group_train = load_dataset(
        config.BASE_DIR_TRAIN, config.Phase
    )
    patient_data_test, patient_to_group_test = load_dataset(
        config.BASE_DIR_TEST, config.Phase
    )
    print(f"\n  Pacients TRAIN: {len(patient_data_train)}")
    print(f"  Pacients TEST:  {len(patient_data_test)}")

    # ──────────────────────────────────────────────
    # 2. SPLIT TRAIN / VALIDATION
    # ──────────────────────────────────────────────
    train_ids, val_ids, _ = patient_stratified_split(
        patient_data_train, patient_to_group_train
    )
    print(f"\nSplit 80/20 estratificat:")
    print(f"  Train: {len(train_ids)} pacients")
    print(f"  Val:   {len(val_ids)} pacients")

    # ──────────────────────────────────────────────
    # 3. ARRAYS DE SLICES
    # ──────────────────────────────────────────────
    X_train, y_train = build_slice_arrays(
        train_ids, patient_data_train, patient_to_group_train
    )
    X_val, y_val = build_slice_arrays(
        val_ids, patient_data_train, patient_to_group_train
    )
    X_test, y_test = build_slice_arrays(
        list(patient_data_test.keys()), patient_data_test, patient_to_group_test
    )
    print(f"\n  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    # ──────────────────────────────────────────────
    # 4. OVERSAMPLING
    # ──────────────────────────────────────────────
    X_train, y_train = apply_oversampling(X_train, y_train)

    # ──────────────────────────────────────────────
    # 5. ANÀLISI DE DADES
    # ──────────────────────────────────────────────
    X_all = np.concatenate([X_train, X_val, X_test])
    analyze_global_images(X_all)
    plot_sample_images(X_all)
    plot_global_histogram(X_all)
    analyze_background(X_all)

    # ──────────────────────────────────────────────
    # 6. CODIFICAR ETIQUETES
    # ──────────────────────────────────────────────
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train, y_val, y_test]))

    y_train_enc = label_encoder.transform(y_train)
    y_val_enc   = label_encoder.transform(y_val)
    y_test_enc  = label_encoder.transform(y_test)
    n_classes   = len(label_encoder.classes_)

    print(f"\nClasses: {label_encoder.classes_}")
    print(f"Nombre de classes: {n_classes}")

    train_counts, val_counts, test_counts = analyze_class_balance(
        y_train_enc, y_val_enc, y_test_enc, label_encoder
    )
    plot_class_distribution(
        train_counts, val_counts, test_counts, label_encoder,
        y_train_enc, y_val_enc, y_test_enc
    )

    # ──────────────────────────────────────────────
    # 7. DATALOADERS
    # ──────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc
    )

    # ──────────────────────────────────────────────
    # 8. MODEL
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("MODEL RESNET18")
    print("="*60)

    if USE_PRETRAINED:
        model = load_pretrained_resnet(
            PRETRAINED_MODEL_PATH, n_classes, device,
            intermediate_size=INTERMEDIATE_SIZE
        )
        skip_training = True
    else:
        model = create_resnet_model(n_classes, device,
                                    intermediate_size=INTERMEDIATE_SIZE)
        skip_training = False

    # ──────────────────────────────────────────────
    # 9. ENTRENAMENT EN 2 FASES
    # ──────────────────────────────────────────────
    if not skip_training:
        print("\n" + "="*60)
        print("ENTRENAMENT (2 fases)")
        print("="*60)

        trainer = Trainer(model, device, save_path="best_model_resnet18.pth")
        history = trainer.train(
            train_loader, val_loader,
            num_epochs=config.NUM_EPOCHS,
            epochs_phase1=EPOCHS_PHASE1
        )
        trainer.plot_history()
        trainer.load_best_model()
    else:
        print("\n  Saltant entrenament (model pre-carregat)")

    # ──────────────────────────────────────────────
    # 10. AVALUACIÓ PER PACIENTS
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("AVALUACIÓ PER PACIENTS")
    print("="*60)

    pacient_pred_train = predict_patients(
        model, {pid: patient_data_train[pid] for pid in train_ids},
        patient_to_group_train, label_encoder, device, is_resnet=True
    )
    analyze_patients(pacient_pred_train, label_encoder, "TRAIN")

    pacient_pred_val = predict_patients(
        model, {pid: patient_data_train[pid] for pid in val_ids},
        patient_to_group_train, label_encoder, device, is_resnet=True
    )
    analyze_patients(pacient_pred_val, label_encoder, "VALIDACIÓ")

    pacient_pred_test = predict_patients(
        model, patient_data_test, patient_to_group_test,
        label_encoder, device, is_resnet=True
    )
    analyze_patients(pacient_pred_test, label_encoder, "TEST")

    # ──────────────────────────────────────────────
    # 11. AVALUACIÓ PER SLICES
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("AVALUACIÓ PER SLICES")
    print("="*60)

    val_tf = get_val_transforms()

    def make_loader(X, y_enc):
        ds = SliceDataset(X, y_enc, transform=val_tf)
        return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    y_pred_train = predict_model(model, make_loader(X_train, y_train_enc), device)
    y_pred_val   = predict_model(model, make_loader(X_val,   y_val_enc),   device)
    y_pred_test  = predict_model(model, make_loader(X_test,  y_test_enc),  device)

    train_acc = evaluate_slices(y_train_enc, y_pred_train, label_encoder, "TRAIN")
    val_acc   = evaluate_slices(y_val_enc,   y_pred_val,   label_encoder, "VALIDACIÓ")
    test_acc  = evaluate_slices(y_test_enc,  y_pred_test,  label_encoder, "TEST")

    # ──────────────────────────────────────────────
    # 12. RESUM FINAL
    # ──────────────────────────────────────────────
    def calc_patient_acc(pred_dict):
        true_l = [info['group_encoded'] for info in pred_dict.values()]
        pred_l = [np.bincount(info['y_pred']).argmax() for info in pred_dict.values()]
        return (np.array(true_l) == np.array(pred_l)).mean()

    tr_p  = calc_patient_acc(pacient_pred_train)
    val_p = calc_patient_acc(pacient_pred_val)
    te_p  = calc_patient_acc(pacient_pred_test)

    print("\n" + "="*60)
    print("RESUM FINAL — RESNET18")
    print("="*60)
    print(f"\n{'':20} {'SLICE':>8}  {'PACIENT':>8}")
    print(f"  {'TRAIN':20} {train_acc:8.4f}  {tr_p:8.4f}")
    print(f"  {'VALIDACIÓ':20} {val_acc:8.4f}  {val_p:8.4f}")
    print(f"  {'TEST':20} {test_acc:8.4f}  {te_p:8.4f}")
    print("="*60)


if __name__ == "__main__":
    main()