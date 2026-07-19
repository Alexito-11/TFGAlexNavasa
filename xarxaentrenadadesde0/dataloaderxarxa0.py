# data_loader.py
# tot el pipeline de dades: llegir els volums nifti de l'ACDC, retallar en slices 2D,
# fer el split per pacient, oversampling i preparar els dataloaders per la CNN

import re
import numpy as np
import cv2
import nibabel as nib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


# TRANSFORMS PER CNN DES DE ZERO
def get_train_transforms():
    # augmentation suau nomes per train: flip, rotacio petita i translacio petita
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ])


def get_val_transforms():
    # val/test sense augmentation, nomes passar a tensor
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])


# DATASET
class SliceDataset(Dataset):
    """
    Dataset de slices cardíaques per CNN des de zero.
    Converteix cada slice (H, W) a tensor (3, H, W) replicant el canal.
    """

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]

        # normalitzo a 0-255 per poder-ho tractar com imatge normal (uint8)
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-8:
            img_u8 = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_u8 = np.zeros_like(img, dtype=np.uint8)

        # replico el canal 3 cops perque les transforms/resnet esperen RGB
        img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)

        if self.transform is not None:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = torch.tensor(
                img_rgb.transpose(2, 0, 1), dtype=torch.float32
            ) / 255.0

        label = torch.tensor(self.y[idx], dtype=torch.long)
        return img_tensor, label


# LECTURA DADES
def read_group_from_info(info_path: Path):
    # llegeixo el grup (patologia) del fitxer Info.cfg de cada pacient
    with open(info_path, "r") as f:
        for line in f:
            if "Group" in line:
                return line.split(":")[1].strip()
    return None


def choose_frame_gt(patient_dir: Path, phase: str):
    # busco els frames amb ground truth d'aquest pacient i em quedo amb ED o ES
    # segons la fase (ED = primer frame, ES = ultim)
    pid = patient_dir.name
    frames = list(patient_dir.glob(f"{pid}_frame*_gt.nii.gz"))
    if not frames:
        return None

    def get_frame_idx(p):
        m = re.search(r"frame(\d+)_gt", p.name)
        return int(m.group(1)) if m else 0

    frames_sorted = sorted(frames, key=get_frame_idx)
    return frames_sorted[0] if phase == "ED" else frames_sorted[-1]


def process_patient_volume(vol, patient_id=""):
    # trec les slices 2D del volum 3D, les redimensiono a target_size,
    # descarto les buides/soroll (variancia baixa) i normalitzo (z-score)
    patient_slices = []
    discarded_slices = 0
    target_size = config.TARGET_SIZE

    if vol.ndim == 3:
        for slice_idx in range(vol.shape[2]):
            img2d = vol[:, :, slice_idx].astype(np.float32)
            img2d_resized = cv2.resize(img2d, target_size, interpolation=cv2.INTER_AREA)

            if config.FILTER_EMPTY_SLICES and img2d_resized.var() < config.VAR_THRESHOLD:
                discarded_slices += 1
                continue

            img2d_resized = (img2d_resized - img2d_resized.mean()) / (img2d_resized.std() + 1e-8)
            patient_slices.append(img2d_resized.astype(np.float32))

        print(f"    {len(patient_slices)} slices vàlides ({discarded_slices} descartades)")
    else:
        # cas rar: volum ja es 2D directament
        img2d = vol.astype(np.float32)
        img2d_resized = cv2.resize(img2d, target_size, interpolation=cv2.INTER_AREA)
        img2d_resized = (img2d_resized - img2d_resized.mean()) / (img2d_resized.std() + 1e-8)
        patient_slices.append(img2d_resized.astype(np.float32))

    return patient_slices, discarded_slices


def load_dataset(base_dir, phase):
    # recorre totes les carpetes de pacients dins base_dir, llegeix el grup,
    # tria el frame gt segons la fase i processa el volum en slices
    patient_data = {}
    patient_to_group = {}
    total_discarded = 0

    print(f"\nProcessant pacients de {base_dir}")

    for patient_dir in sorted(base_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        info_path = patient_dir / "Info.cfg"
        if not info_path.exists():
            continue

        group = read_group_from_info(info_path)
        if group is None:
            continue

        nii_path = choose_frame_gt(patient_dir, phase)
        if nii_path is None:
            continue

        print(f"  {patient_dir.name} → {group}")
        vol = nib.load(str(nii_path)).get_fdata()
        patient_slices, discarded = process_patient_volume(vol, patient_dir.name)
        total_discarded += discarded

        if patient_slices:
            patient_data[patient_dir.name] = np.stack(patient_slices, axis=0)
            patient_to_group[patient_dir.name] = group
        else:
            print(f"  ATENCIÓ: {patient_dir.name} sense slices vàlides!")

    print(f"\nTotal slices descartades: {total_discarded}")
    return patient_data, patient_to_group


# SPLIT I PREPARACIÓ
def patient_stratified_split(patient_data, patient_to_group, test_size=0.2):
    # split train/val fet a nivell de pacient (no de slice) per no tenir
    # slices del mateix pacient repartides entre train i val
    patient_ids = np.array(list(patient_data.keys()))
    patient_groups = np.array([patient_to_group[pid] for pid in patient_ids])

    le = LabelEncoder()
    y_patients = le.fit_transform(patient_groups)

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=config.SEED
    )
    train_idx, val_idx = next(sss.split(patient_ids, y_patients))

    train_ids = patient_ids[train_idx]
    val_ids = patient_ids[val_idx]

    return train_ids, val_ids, le


def build_slice_arrays(patient_ids, patient_data, patient_to_group):
    # aplano les slices de tots els pacients d'un split en un unic array X, y
    X_list, y_list = [], []

    for pid in patient_ids:
        for sl in patient_data[pid]:
            X_list.append(sl)
            y_list.append(patient_to_group[pid])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y


def apply_oversampling(X_train, y_train):
    # oversampling amb reemplaçament per igualar totes les classes a la majoritaria
    print("\n" + "=" * 60)
    print("OVERSAMPLING")
    print("=" * 60)

    classes, counts = np.unique(y_train, return_counts=True)
    majority_count = max(counts)

    print("Distribució ABANS:")
    for cls, cnt in zip(classes, counts):
        print(f"  {cls}: {cnt}")

    X_balanced, y_balanced = [], []

    for cls in classes:
        mask = y_train == cls
        X_cls, y_cls = X_train[mask], y_train[mask]

        if len(X_cls) < majority_count:
            X_res, y_res = resample(
                X_cls, y_cls,
                replace=True,
                n_samples=majority_count,
                random_state=config.SEED
            )
            X_balanced.append(X_res)
            y_balanced.append(y_res)
        else:
            X_balanced.append(X_cls)
            y_balanced.append(y_cls)

    X_balanced = np.vstack(X_balanced)
    y_balanced = np.concatenate(y_balanced)

    print("\nDistribució DESPRÉS:")
    classes2, counts2 = np.unique(y_balanced, return_counts=True)
    for cls, cnt in zip(classes2, counts2):
        print(f"  {cls}: {cnt}")

    print(f"\nTotal: {len(X_train)} → {len(X_balanced)} slices")
    return X_balanced, y_balanced


# DATALOADERS
def create_dataloaders(X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc):
    """
    Crea DataLoaders per la CNN des de zero.
    - Train: augmentation suau
    - Val/Test: sense augmentation
    """
    train_ds = SliceDataset(X_train, y_train_enc, transform=get_train_transforms())
    val_ds   = SliceDataset(X_val,   y_val_enc,   transform=get_val_transforms())
    test_ds  = SliceDataset(X_test,  y_test_enc,  transform=get_val_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print("\nDataLoaders creats per CNN des de zero:")
    print(f"  Train: {len(train_ds):5d} slices ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_ds):5d} slices ({len(val_loader)} batches)")
    print(f"  Test:  {len(test_ds):5d} slices ({len(test_loader)} batches)")

    return train_loader, val_loader, test_loader
