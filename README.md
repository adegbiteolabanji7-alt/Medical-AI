Medical AI: 3D Spleen Segmentation
🩺 Project Overview
This repository contains a deep learning workflow for the automated segmentation of the spleen from 3D CT scans. This is a critical task in medical imaging for volumetric analysis and surgical planning.
Goal: Accurately delineate the spleen boundaries in 3D medical images.
Framework: Python-based implementation utilizing the MONAI (Medical Open Network for AI) framework and PyTorch.
Dataset: Utilizes the "Spleen" dataset from the Medical Segmentation Decathlon.
🛠️ Technical Stack
Language: Python 3.11.
Environment: Managed via Anaconda/Miniconda in a dedicated medai environment.
Tools: Jupyter Notebooks, VS Code, and Git.
📂 Repository Structure
spleen_segmentation_3d.ipynb: The main training and evaluation pipeline.
.gitignore: Configured to exclude large medical data files (.nii.gz) to keep the repository lightweight.
🚀 Future Work
[ ] Experiment with different U-Net architectures.
[ ] Implement Dice Loss optimization for better accuracy.
[ ] Deploy the model as a simple web-based inference tool.