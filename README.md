# Medical AI — Clinical Imaging Portfolio

A portfolio of applied AI projects in medical imaging, 
built by a diagnostic sonographer training in healthcare AI.

## Projects

### 1. DICOM Router (active)
Automatically routes DICOM medical imaging files into organised 
folders based on the Modality tag (0008,0060).

**Why it matters:** Every hospital AI pipeline starts with data 
ingestion. Correctly routing CT, MRI, and ultrasound files is 
the foundation before any model touches the data.

**Stack:** Python · pydicom · pathlib · logging

### 2. Medical Image Segmentation Pipeline 
End-to-end deep learning pipeline for 3D organ segmentation.
**Stack:** PyTorch · MONAI · Medical Segmentation Decathlon
**Result:** Dice score 0.24 after 50 epochs on Medical Segmentation Decathlon spleen dataset (Google Colab, T4 GPU). Known limitation — score indicates undertrained model; optimisation (longer training, loss function tuning, data augmentation) identified as next step.

### 3. Ultrasound AI for Low-Resource Settings (in progress — MSc dissertation, pending formal approval)

CNN classifier for liver lesion classification (HCC vs hemangioma) on B-mode ultrasound, testing robustness under simulated low-resource imaging degradation. Dataset: SMC-LUD (Nature Scientific Data, 2026).

## About
MSc AI for Healthcare, University of Hull.  
Background: BSc Biochemistry + diagnostic sonography.  
Focus: Clinically-motivated AI tools with regulatory awareness.
