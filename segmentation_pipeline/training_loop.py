"""
training_loop.py
----------------
Training loop for 3D U-Net segmentation.

The outer skeleton that orchestrates everything:
1. Load batch from data loader
2. Zero gradients - must clear before each backward pass
3. Forward pass - data through U-Net, get predicted mask
4. Compute loss - DiceCELoss compares prediction to ground truth
5. Backward pass - PyTorch calculates gradients for all weights
6. Optimizer step - Adam updates weights to reduce loss
7. Repeat for all batches - one full pass = one epoch
8. Validate after each epoch on unseen cases

Why zero gradients before forward pass?
PyTorch accumulates gradients by default. Without zeroing,
gradients from previous batch corrupt current update.

Why DiceCELoss not Dice alone?
Cross Entropy: stable gradients early in training when
predictions are random noise far from ground truth.
Dice Loss: measures volumetric overlap, directly optimises
the clinical metric. Combined: stability + clinical relevance.

Why model.train() and model.eval()?
BatchNorm behaves differently in training versus inference.
Training: uses current batch statistics.
Inference: uses accumulated running statistics.
Wrong mode gives incorrect predictions.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import sys
sys.path.insert(0, 'segmentation_pipeline')
from data_loader import train_loader, val_loader
from unet_model import UNet3D

# Device - GPU if available, CPU otherwise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Model
model = UNet3D(in_channels=1, out_channels=1).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss - Dice loss for segmentation
loss_fn = DiceLoss(sigmoid=True)

# Optimizer - Adam adapts learning rate per parameter
optimizer = Adam(model.parameters(), lr=1e-4)

# Metric
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Training
num_epochs = 5
best_dice = 0.0

print(f"\nStarting training for {num_epochs} epochs...\n")

for epoch in range(num_epochs):

    # Training phase
    model.train()
    epoch_loss = 0.0

    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()          # Step 2: clear gradients
        predictions = model(images)    # Step 3: forward pass
        loss = loss_fn(predictions, labels)  # Step 4: compute loss
        loss.backward()                # Step 5: backward pass
        optimizer.step()               # Step 6: update weights

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    # Validation phase
    model.eval()
    dice_metric.reset()

    with torch.no_grad():
        for val_batch in val_loader:
            val_images = val_batch["image"].to(device)
            val_labels = val_batch["label"].to(device)
            val_preds = model(val_images)
            val_preds = (val_preds > 0.5).float()
            dice_metric(y_pred=val_preds, y=val_labels)

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Loss: {avg_loss:.4f} | "
          f"Val Dice: {mean_dice:.4f}")

    if mean_dice > best_dice:
        best_dice = mean_dice
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  New best model saved. Dice: {best_dice:.4f}")

print(f"\nTraining complete. Best Dice: {best_dice:.4f}")