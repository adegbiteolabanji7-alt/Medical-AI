"""
first_monai.py
-----------------
First contact with MONAI.

MONAI (Medical Open Network for AI) is a PyTorch-based framework for deep learning in healthcare imaging. It provides a set of tools and libraries for building, training, and deploying deep learning models in medical imaging applications.
It handles things that make medical data from natural images:
- NIfTI and DICOM formats
- 3D volumetric data
- Spatially aware transformations that preserve clinical meaning.
-Patch based training for large volumes 
Why MONAI over PyTorch?
Loading a CT scan, resampling to consistent voxel spacing,normalizing intensity, extracting patches, and augmenting whike respoecting the spatial context is a lot of work. MONAI provides built-in tools to handle these tasks efficiently, allowing you to focus on model development rather than data preprocessing.
what would take weeks of writing from scratch in pyTorch, could be done in ten lines with MONAI. the logic is the same, the engineering is just done for you


"""
import monai 
from monai.transforms import(
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    ToTensord,

)
from monai.data import Dataset, DataLoader
from monai.utils import first
import numpy as np

print(f"MONAI version: {monai.__version__}")

#--- Understanding MONAI transforms---
# MONAI uses dictionary-based transforms.
# Your data is always a dictionary, with keys like "image" and "labels"
# Every transforms knows which key to operate on.
# this keeps image and its segmentation mask in sync automatically.

# Why does it matter?
# If you flip an image for augmentation, the segmentation mask mjust flip identically. MONAI handles this automatically.
# Raw PyTorch does not.

# A transform pipeline for loading a 3D medical image
transform_pipeline = Compose([
    LoadImaged(keys=["image"]), # Load the image, NIfTI file
    EnsureChannelFirstd(keys=["image"]), # Ensure the image has a channel dimension: ( H, W, D)
    ScaleIntensityRanged( #Normalize CT intensity values
        keys=["image"],
        a_min=-57, # Minimum Hounsfield units for CT scans(soft tissue)
        a_max=164, # Maximum Hounsfield units for CT scans(soft tissue)
        b_min=0.0, # Scale to 0
        b_max=1.0, # Scale to 1
        clip=True # clip values outside the range to the specified min and max
    ),
    Resized(keys=["image"],
             spatial_size=(96, 96, 96)), # Resample to a standard patch size
    ToTensord(keys=["image"]), # Convert to PyTorch tensor
])
print("\nMONAI transform pipeline created successfully.")
print(transform_pipeline)
# --- Hounsfield Units (HU) ---
# CT scanners measure X-ray attenuation in Hounsfield Units.
# Air = -1000 HU, Water = 0 HU, Soft tissue = -57 to 164 HU
# Bone = 400 to 1000 HU
# We clip to soft tissue range because we are segmenting organs,
# not bone. Values outside this range are irrelevant and
# distort the normalisation.
# As a sonographer you know tissue contrast — HU is CT's equivalent
# of echogenicity. Different tissues, different values.

print("\nHounsfield Units (HU) are a standardized scale for measuring X-ray attenuation in CT scans. They help differentiate between different tissue types based on their density. For example, air has a value of -1000 HU, water is 0 HU, and soft tissues typically range from -57 to 164 HU. Bone can have values from 400 to 1000 HU. By clipping the intensity values to the soft tissue range, we focus on the relevant information for organ segmentation while ignoring irrelevant values that could distort the normalization process.")
print("AIR:    -1000 HU")
print("FAT:    -100  HU")
print("Soft tissue: -57 to 164 HU")
print("Bone:   400 to 1000 HU")

print("\nMonai is ready. Next: load real medical data.")

