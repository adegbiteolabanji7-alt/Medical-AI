"""
first_monai.py
-------------   
First contact with MONAI for 3D medical image segmentation..

MONAI is built on pytorch specifically for medical imaging,
It handles the things that make medical data different from natural images:
- 3D volumes instead of 2D images
- NIfTI and DICOM formats instead of JPEG/PNG
- Spatially-aware transformations that preserve clinical meaning.
- Patch based training for large volumes.

Why MONAI over raw pytorch?
loading a CT scan, resampling to consistent voxels spacing, normalizing intensity, extracting patches and augmenting while respecting 3D geometry would take weeks from scratch. MONAI does it in ten lines.

Why dictionary-based transformss?
When you flip an image for augumentation,the segmentation mask must flip identically. Dictionary transforms keep image and mask in sync automatically. out of sync means the model trains on wrong data and learns nothing.
"""

import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    ToTensord,
)

print(f"MONAI version: {monai.__version__}")

# Transform pipeline for loading and preprocessing a 3D medical image and its segmentation mask
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    ToTensord,
)

print(f"MONAI version: {monai.__version__}")

# Transform pipeline for loading a 3D medical image
transform_pipeline = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-57,    # Minimum HU value for soft tissue
        a_max=164,    # Maximum HU value for soft tissue
        b_min=0.0,    # Scale to 0
        b_max=1.0,    # Scale to 1
        clip=True,    # Clip values outside range
    ),
    Resized(keys=["image"], spatial_size=(96, 96, 96)),
    ToTensord(keys=["image"]),
])

print("\nTransforms pipeline created successfully.")

# Hounsfield Units - CT tissue density scale
# Neural networks work best with values between 0 and 1
# We clip to soft tissue range because we segment organs not bone
print("\nHounsfield Unit ranges:")
print("Air:          -1000 HU")
print("Fat:           -100 HU")
print("Soft tissue:   -57 to 164 HU  <- our range")
print("Bone:          400 to 1000 HU")

print("\nMONAI is ready. Next: load real medical data.")