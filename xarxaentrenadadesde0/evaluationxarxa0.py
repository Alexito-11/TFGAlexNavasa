# evaluation.py
# funcions per avaluar els models (resnet i cnn), tant per slice com per pacient (vot majoritari)

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader


def predict_model(model, data_loader, device):
    """Fa prediccions donant un DataLoader (amb normalització ImageNet correcta)"""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in data_loader:
            xb = xb.to(device)
            outputs = model(xb)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)


def evaluate_slices(y_true, y_pred, label_encoder, dataset_name=""):
    """Avaluació a nivell de slice"""
    print(f"\n{'='*60}")
    print(f"ANALISI PER SLICES - {dataset_name}")
    print(f"{'='*60}")

    print(f"\n=== Classification Report ({dataset_name}) ===")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    print(f"\n=== Confusion Matrix ({dataset_name}) ===")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=label_encoder.classes_,
        cmap='Blues'
    )
    plt.title(f"Matriu de confusió ({dataset_name} - per slice)")
    plt.tight_layout()
    plt.show()

    acc = (y_pred == y_true).mean()
    print(f"Precisio en {dataset_name}: {acc:.3f}")
    return acc


def predict_patients(model, patient_data, patient_to_group, label_encoder, device,
                     is_resnet=False):
    """
    Prediccions a nivell de pacient.
    Usa sempre SliceDataset + get_val_transforms(), el mateix pipeline
    amb què s'entrenen totes dues xarxes (CNN i ResNet18).
    """
    # import aqui dins per no crear import circular amb data_loader
    from data_loader import SliceDataset, get_val_transforms
    import config

    pacient_predictions = {}
    val_transform = get_val_transforms()   # SEMPRE, per a les dues xarxes (CNN i ResNet)

    model.eval()
    with torch.no_grad():
        for pacient_id, slices in patient_data.items():
            group = patient_to_group[pacient_id]

            # dataset "de mentida" nomes per poder passar les slices d'aquest pacient
            # pel mateix pipeline de transform que en el train
            dummy_labels = np.zeros(len(slices), dtype=np.int64)
            ds = SliceDataset(slices, dummy_labels, transform=val_transform)
            dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

            preds_list = []
            for xb, _ in dl:
                xb = xb.to(device)
                outputs = model(xb)
                _, p = torch.max(outputs, 1)
                preds_list.append(p.cpu().numpy())

            y_pred_pacient = np.concatenate(preds_list)
            y_true_pacient = label_encoder.transform([group] * len(slices))

            pacient_predictions[pacient_id] = {
                'group':         group,
                'group_encoded': label_encoder.transform([group])[0],
                'num_slices':    len(slices),
                'y_true':        y_true_pacient,
                'y_pred':        y_pred_pacient,
                'accuracy':      (y_pred_pacient == y_true_pacient).mean()
            }

    return pacient_predictions


def analyze_patients(pacient_predictions, label_encoder, dataset_name=""):
    """Anàlisi a nivell de pacient"""
    if not pacient_predictions:
        print(f"\nNo hi ha pacients a {dataset_name}")
        return

    n_classes = len(label_encoder.classes_)

    print(f"\n{'='*60}")
    print(f"ANALISI PER PACIENTS - {dataset_name}")
    print(f"{'='*60}")
    print(f"Total pacients: {len(pacient_predictions)}")

    # agrupo els pacients per classe real, per treure estadistiques despres
    pacients_per_classe = {}
    for pid, info in pacient_predictions.items():
        cls = info['group']
        pacients_per_classe.setdefault(cls, []).append(pid)

    print(f"\nDISTRIBUCIO PER CLASSE:")
    for cls, pids in pacients_per_classe.items():
        print(f"  {cls}: {len(pids)} pacients")

    # accuracy per pacient (slice a slice, sense vot encara), classe per classe
    print(f"\n{'='*60}")
    print(f"ESTADISTIQUES PER CLASSE ({dataset_name})")
    print(f"{'='*60}")

    for cls, pids in pacients_per_classe.items():
        accs = []
        print(f"\n{cls}:")
        print(f"  {'Pacient':<12} {'Slices':<8} {'Encerts':<10} {'Errors':<10} {'Accuracy':<8}")
        print(f"  " + "-"*48)
        for pid in sorted(pids):
            info      = pacient_predictions[pid]
            acc       = info['accuracy']
            accs.append(acc)
            correctes = (info['y_pred'] == info['y_true']).sum()
            errors    = len(info['y_pred']) - correctes
            print(f"  {pid:<12} {info['num_slices']:<8} {correctes:<10} {errors:<10} {acc:<8.3f}")
        print(f"  " + "-"*48)
        print(f"  Mitjana: {np.mean(accs):.3f}  Desv: {np.std(accs):.3f}")

    # aqui ja fem vot majoritari: cada pacient es queda amb la classe mes votada entre les seves slices
    print(f"\n{'='*60}")
    print(f"MATRIU DE CONFUSIO PER PACIENTS - {dataset_name} (VOT MAJORITARI)")
    print(f"{'='*60}")

    pacient_true, pacient_pred, pacient_classes = [], [], []
    for pid in sorted(pacient_predictions.keys()):
        info = pacient_predictions[pid]
        pacient_true.append(info['group_encoded'])
        unique_p, counts = np.unique(info['y_pred'], return_counts=True)
        pacient_pred.append(unique_p[np.argmax(counts)])
        pacient_classes.append(info['group'])

    cm = confusion_matrix(pacient_true, pacient_pred, labels=range(n_classes))
    print(cm)

    print(f"\nAccuracy per classe (vot majoritari):")
    for i, cls in enumerate(label_encoder.classes_):
        mask = np.array(pacient_classes) == cls
        if np.sum(mask) > 0:
            acc_cls = (np.array(pacient_true)[mask] == np.array(pacient_pred)[mask]).mean()
            print(f"  {cls}: {acc_cls:.3f} ({np.sum(mask)} pacients)")

    ConfusionMatrixDisplay.from_predictions(
        pacient_true, pacient_pred,
        display_labels=label_encoder.classes_,
        cmap='Blues'
    )
    plt.title(f"Matriu de confusió per pacient ({dataset_name} - vot majoritari)")
    plt.tight_layout()
    plt.show()

    acc_global = (np.array(pacient_true) == np.array(pacient_pred)).mean()
    print(f"\nPrecisio global per pacient ({dataset_name}): {acc_global:.4f}")
    return acc_global
