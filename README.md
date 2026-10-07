# Medical AI. Clinical Imaging Portfolio

A portfolio of applied AI projects in medical imaging, built by a diagnostic sonographer training in healthcare AI.

## Projects

### 1. DICOM Router (active)

Automatically routes DICOM medical imaging files into organised folders based on the Modality tag (0008,0060).

**Why it matters:** Every hospital AI pipeline starts with data ingestion. Correctly routing CT, MRI, and ultrasound files is the foundation before any model touches the data.

**Stack:** Python · pydicom · pathlib · logging

### 2. Medical Image Segmentation Pipeline

End-to-end deep learning pipeline for 3D organ segmentation.

**Stack:** PyTorch · MONAI · Medical Segmentation Decathlon

**Result:** Dice score 0.24 after 50 epochs on the Medical Segmentation Decathlon spleen dataset (Google Colab, T4 GPU). Training was stopped while Dice was still improving epoch over epoch, so the model was very likely undertrained rather than fundamentally broken. I haven't yet gone back to confirm that against a longer run, which is the next step: resume training for significantly more epochs, and if Dice still plateaus low, check next whether the loss function (currently plain cross-entropy) needs to move to a Dice-aware or combined loss, since segmentation with class imbalance between organ and background often needs that regardless of training length.

### 3. Ultrasound AI for Low-Resource Settings (MSc dissertation, in progress)

CNN classifier for liver lesion classification (HCC vs hemangioma) on B-mode ultrasound, testing robustness under simulated low-resource imaging degradation. Dataset: SMC-LUD (Nature Scientific Data, 2026). Supervised by Eva Sousa, University of Hull.

## Repository Structure

```
dicom_router/           # DICOM file routing by modality
segmentation_pipeline/  # 3D U-Net segmentation (MONAI, PyTorch)
pytorch_fundamentals/   # Core Python/PyTorch practice and learning
data/                   # Local data (gitignored, not tracked)
```

## About

MSc AI for Healthcare, University of Hull. Background: BSc Biochemistry + diagnostic sonography. Focus: Clinically-motivated AI tools with regulatory awareness.
