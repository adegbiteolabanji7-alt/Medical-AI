# Medical Image Segmentation Pipeline

## Clinical Context

In clinical practice, measuring organ volumes like the spleen is 
essential for tracking infections, blood disorders, and trauma. 
Currently this requires a clinician to manually delineate the 
spleen slice by slice across a 3D CT volume, time-consuming and 
subject to inter-observer variability.

As a diagnostic sonographer I understand the challenge of 
identifying organ boundaries when contrast is low and image 
quality is variable. This pipeline automates spleen segmentation 
using deep learning, providing objective and reproducible 
volumetric measurements.

## Connection to Dissertation

This spleen segmentation project is a technical foundation for 
my MSc dissertation on platelet segmentation under low-resource 
imaging conditions. Both tasks share the same core challenge, 
identifying small target structures against a dominant background. 
The pipeline architecture, training loop, and evaluation metrics 
developed here transfer directly to the dissertation work.

## Architecture

I chose U-Net because it was designed specifically for biomedical 
image segmentation and remains the industry standard for organ 
delineation tasks.

The encoder compresses the input CT volume progressively through 
convolutional layers, learning increasingly abstract features at 
each level. The bottleneck captures the most compressed 
representation, the deepest abstract understanding of the image. 
The decoder expands back to the original spatial resolution using 
those features. Skip connections pass feature maps directly from 
each encoder level to the corresponding decoder level, preserving 
spatial detail that pooling would otherwise discard. Without skip 
connections the decoder cannot accurately reconstruct precise 
organ boundaries.

## Dataset

Medical Segmentation Decathlon, Spleen Task.
82 abdominal CT scans with manual segmentation masks annotated 
by clinical experts. Significant variation in spleen size across 
cases, some normal, some massively enlarged due to pathology. 
This variability makes the task clinically realistic and 
technically challenging.

## Loss Function

The spleen occupies a small fraction of the total CT volume. 
A naive model predicting background for every voxel would achieve 
high accuracy while completely missing the spleen.

I used DiceCELoss, a combination of Dice loss and Cross Entropy.

Cross Entropy provides stable gradients at the individual voxel 
level, particularly important in early training when predictions 
are far from the ground truth. Dice loss measures the volumetric 
overlap between predicted and ground truth masks, directly 
optimising the clinical metric we care about. Combined, they give 
early training stability from Cross Entropy and clinical relevance 
from Dice loss.

## Results

Phase 1, Pipeline validation (16 cases, 50 epochs)
Best Dice: 0.2425

A Dice score of 0.24 confirms the pipeline is functioning 
correctly. The low score is expected, 16 cases is insufficient 
for the model to learn generalised spleen anatomy. This run 
validated the data loading, normalisation, patch extraction, 
training loop, and evaluation code.

Phase 2, Full dataset training (82 cases, in progress)
Training on the complete dataset. Dice improving consistently 
epoch by epoch. Results to be updated on completion.

## Stack

PyTorch, MONAI, nibabel, Google Colab (T4 GPU)

## Regulatory Context

In a clinical deployment context, a spleen segmentation tool 
intended to inform clinical decisions would qualify as a medical 
device under UK MDR 2002. This would require clinical validation, 
regulatory submission, and a human-in-the-loop workflow where 
a clinician reviews and approves all model predictions before 
they influence patient management.