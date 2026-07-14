# data_loader.py
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


# ─── Transformacions ImageNet ──────────────────────────────────────────────────
# ResNet18 va ser pre-entrenada amb imatges RGB normalitzades amb mean/std
# d'ImageNet. Hem de replicar exactament aquesta normalització, altrament
# els pesos pre-entrenats no funcionaran correctament.

def get_train_transforms():
    """Train: data augmentation + normalització ImageNet"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # ← afegir
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),  # ← afegir
        transforms.RandomAutocontrast(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_val_transforms():
    """Val/Test: només normalització ImageNet, sense augmentation"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


class SliceDataset(Dataset):
    """
    Dataset de slices cardíaques per a ResNet18.
    Cada imatge en escala de grisos es replica a 3 canals (RGB fals)
    perquè ResNet espera entrada RGB.
    """

    def __init__(self, X, y, transform=None):
        self.X         = X          # (N, H, W) float32
        self.y         = y          # (N,) int
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]  # (H, W) float32

        # Escalar a [0, 255] uint8 per ToPILImage
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-8:
            img_u8 = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_u8 = np.zeros_like(img, dtype=np.uint8)

        # Replicar a 3 canals: (H, W) → (H, W, 3)
        img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)

        if self.transform:
            tensor = self.transform(img_rgb)   # (3, H, W) normalitzat ImageNet
        else:
            tensor = torch.tensor(
                img_rgb.transpose(2, 0, 1), dtype=torch.float32
            ) / 255.0

        label = torch.tensor(self.y[idx], dtype=torch.long)
        return tensor, label


# ─── Càrrega i processament de dades ──────────────────────────────────────────

def read_group_from_info(info_path: Path):
    with open(info_path, "r") as f:
        for line in f:
            if "Group" in line:
                return line.split(":")[1].strip()
    return None


def choose_frame_gt(patient_dir: Path, phase: str):
    pid    = patient_dir.name
    frames = list(patient_dir.glob(f"{pid}_frame*_gt.nii.gz"))
    if not frames:
        return None

    def get_frame_idx(p):
        m = re.search(r"frame(\d+)_gt", p.name)
        return int(m.group(1)) if m else 0

    frames_sorted = sorted(frames, key=get_frame_idx)
    return frames_sorted[0] if phase == "ED" else frames_sorted[-1]


def process_patient_volume(vol, patient_id=""):
    pacient_slices    = []
    slices_descartades = 0
    target_size       = config.TARGET_SIZE  # (224, 224)

    if vol.ndim == 3:
        for slice_idx in range(vol.shape[2]):
            img2d         = vol[:, :, slice_idx].astype(np.float32)
            img2d_resized = cv2.resize(img2d, target_size, interpolation=cv2.INTER_AREA)

            if config.FILTER_EMPTY_SLICES and img2d_resized.var() < config.VAR_THRESHOLD:
                slices_descartades += 1
                continue

            # Normalització per slice
            img2d_resized = (img2d_resized - img2d_resized.mean()) / (img2d_resized.std() + 1e-8)
            pacient_slices.append(img2d_resized.astype(np.float32))

        print(f"    {len(pacient_slices)} slices vàlides ({slices_descartades} descartades)")
    else:
        img2d         = vol.astype(np.float32)
        img2d_resized = cv2.resize(img2d, target_size, interpolation=cv2.INTER_AREA)
        img2d_resized = (img2d_resized - img2d_resized.mean()) / (img2d_resized.std() + 1e-8)
        pacient_slices.append(img2d_resized.astype(np.float32))

    return pacient_slices, slices_descartades


def load_dataset(base_dir, phase):
    patient_data     = {}
    patient_to_group = {}
    total_desc       = 0

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
        vol                    = nib.load(str(nii_path)).get_fdata()
        pacient_slices, desc   = process_patient_volume(vol, patient_dir.name)
        total_desc            += desc

        if pacient_slices:
            patient_data[patient_dir.name]     = np.stack(pacient_slices, axis=0)
            patient_to_group[patient_dir.name] = group
        else:
            print(f"  ATENCIÓ: {patient_dir.name} sense slices vàlides!")

    print(f"\nTotal slices descartades: {total_desc}")
    return patient_data, patient_to_group


def patient_stratified_split(patient_data, patient_to_group, test_size=0.2):
    patient_ids    = np.array(list(patient_data.keys()))
    patient_groups = np.array([patient_to_group[pid] for pid in patient_ids])

    le  = LabelEncoder()
    enc = le.fit_transform(patient_groups)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=config.SEED)
    train_idx, val_idx = next(sss.split(patient_ids, enc))

    return patient_ids[train_idx], patient_ids[val_idx], le


def build_slice_arrays(patient_ids, patient_data, patient_to_group):
    X_list, y_list = [], []
    for pid in patient_ids:
        for sl in patient_data[pid]:
            X_list.append(sl)
            y_list.append(patient_to_group[pid])
    return np.stack(X_list, axis=0), np.array(y_list)


def apply_oversampling(X_train, y_train):
    print("\n" + "="*60)
    print("OVERSAMPLING")
    print("="*60)

    classes, counts  = np.unique(y_train, return_counts=True)
    majority_count   = max(counts)

    print("Distribució ABANS:")
    for cls, cnt in zip(classes, counts):
        print(f"  {cls}: {cnt}")

    X_bal, y_bal = [], []
    for cls in classes:
        mask = y_train == cls
        Xc, yc = X_train[mask], y_train[mask]
        if len(Xc) < majority_count:
            Xr, yr = resample(Xc, yc, replace=True,
                              n_samples=majority_count, random_state=config.SEED)
            X_bal.append(Xr); y_bal.append(yr)
        else:
            X_bal.append(Xc); y_bal.append(yc)

    X_bal = np.vstack(X_bal)
    y_bal = np.concatenate(y_bal)
    print(f"\nTotal: {len(X_train)} → {len(X_bal)} slices")
    return X_bal, y_bal


def create_dataloaders(X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc):
    """
    Crea DataLoaders amb transformacions correctes per ResNet18:
    - Train: augmentation + normalització ImageNet
    - Val/Test: només normalització ImageNet
    Les imatges ja surten com a tensors (3, 224, 224) normalitzats.
    """
    train_ds = SliceDataset(X_train, y_train_enc, transform=get_train_transforms())
    val_ds   = SliceDataset(X_val,   y_val_enc,   transform=get_val_transforms())
    test_ds  = SliceDataset(X_test,  y_test_enc,  transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"\nDataLoaders creats (imatges 3x224x224, normalització ImageNet):")
    print(f"  Train: {len(train_ds):5d} slices  ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_ds):5d} slices  ({len(val_loader)} batches)")
    print(f"  Test:  {len(test_ds):5d} slices  ({len(test_loader)} batches)")

    return train_loader, val_loader, test_loader