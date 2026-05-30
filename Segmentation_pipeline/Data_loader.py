"""
data_loader.py
--------------
Loading and preprocessing 3D medical imaging data with MONAI.

We use synthetic data here for development.
Swap dataset_dir to point to real Medical Segmentation Decathlon
data when available. The pipeline is identical.

Why synthetic data for development?
Real medical volumes are 1-2GB per dataset. Waiting for downloads
during development wastes time. Synthetic data lets you verify
the pipeline logic is correct before committing to real data.
Professional pipelines always have this mode.
"""

import numpy as np
import torch
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ToTensord,
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.utils import first
import tempfile
import os
import nibabel as nib

print("Setting up synthetic medical imaging dataset...\n")

# --- Create synthetic NIfTI volumes ---
# Simulates CT scan volumes with segmentation masks
# Real data would come from Medical Segmentation Decathlon

def create_synthetic_volume(shape=(128, 128, 64)):
    """
    Create a synthetic CT-like volume and segmentation mask.
    
    The volume simulates soft tissue HU values.
    The mask simulates a single organ segmentation.
    
    In real data:
    - Volume comes from CT scanner as DICOM, converted to NIfTI
    - Mask comes from radiologist manual annotation
    """
    # Synthetic CT intensities in HU range
    volume = np.random.randint(-100, 200, shape).astype(np.float32)
    
    # Synthetic organ mask - sphere in centre of volume
    mask = np.zeros(shape, dtype=np.uint8)
    centre = [s // 2 for s in shape]
    radius = min(shape) // 4
    
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if (x - centre[0])**2 + \
                   (y - centre[1])**2 + \
                   (z - centre[2])**2 < radius**2:
                    mask[x, y, z] = 1
    
    return volume, mask


# Create temporary directory for synthetic data
temp_dir = tempfile.mkdtemp()
print(f"Creating synthetic volumes in: {temp_dir}")

# Generate 4 synthetic cases (train) + 2 (validation)
data_files = []
for i in range(6):
    volume, mask = create_synthetic_volume(shape=(64, 64, 32))
    
    vol_path = os.path.join(temp_dir, f"volume_{i}.nii.gz")
    mask_path = os.path.join(temp_dir, f"mask_{i}.nii.gz")
    
    nib.save(nib.Nifti1Image(volume, np.eye(4)), vol_path)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    
    data_files.append({"image": vol_path, "label": mask_path})
    print(f"Created case {i+1}: volume {volume.shape}, "
          f"mask unique values: {np.unique(mask)}")

# Split into train and validation
train_files = data_files[:4]
val_files = data_files[4:]

print(f"\nTraining cases: {len(train_files)}")
print(f"Validation cases: {len(val_files)}")

# --- Transform pipeline ---
# Two pipelines: one for training (with augmentation)
# one for validation (no augmentation - we want consistent evaluation)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-57, a_max=164,
        b_min=0.0, b_max=1.0,
        clip=True,
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-57, a_max=164,
        b_min=0.0, b_max=1.0,
        clip=True,
    ),
    ToTensord(keys=["image", "label"]),
])

print("\nTransform pipelines created:")
print("Training: load -> normalise -> augment -> tensor")
print("Validation: load -> normalise -> tensor (no augmentation)")

# --- Dataset and DataLoader ---
train_dataset = Dataset(data=train_files, transform=train_transforms)
val_dataset = Dataset(data=val_files, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print(f"\nDataset created:")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# --- Inspect one batch ---
print("\nInspecting first training batch...")
batch = first(train_loader)
image = batch["image"]
label = batch["label"]

print(f"Image shape: {image.shape}")
print(f"Label shape: {label.shape}")
print(f"Image dtype: {image.dtype}")
print(f"Image min: {image.min():.3f}, max: {image.max():.3f}")
print(f"Label unique values: {label.unique()}")

print("\nData loader ready.")
print("Swap temp_dir for real dataset path when on wifi.")