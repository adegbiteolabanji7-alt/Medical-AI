"""
training_loop.py
-------------
The training loop for 3D U-Net segmentation.

This is the outer skeleton that orchestrates everything:
- Loads batches from the data loader
- passes them through the U-Net model(forwad pass)
- computes the loss(how wrong the prediction is)
- backpropagates the error to update model weights(calculates the gradients)
- updates the model parameters using an optimizer(step)
- evaluates performance on a validation set periodically(after each epoch)



Why Dice loss and not Cross-Entropy?
Cross entropy treats every vixels equally. in organ segmentation, the background dominates --- 98% of the voxels are not spleen.
Cross entropy would reward a model that predicts all background. Dice loss only cares about the overlap of the target structure.  it directly optimises what we avtually want to measure.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
import sys
sys.path.append("Segmentation_pipeline")
from Unet_model import UNet3D
from Data_loader import train_loader, val_loader

#-----Device-------
# Use GPU if available, otherwise fallback to CPU
# In collab this will be CUDA-- much faster.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

#-----Model-------
model = UNet3D(in_channels=1, out_channels=1).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

#-----Loss function-------
# DiceLoss from MONAI
# sigmoid = True because our models outputs raw values
# we apply sigmoid inside the loss.
loss_fn = DiceLoss(sigmoid=True)

#-----Optimizer-----
# Adam is a popular choice for training deep learning models.
# lr is the learning rate, which controls how much we update the model parameters at each step.
# lr = 1e-4 is a common starting point for training U-Nets on medical images.
# Too high: weight overshoots, training diverges
# Too low: training is very slow, may get stuck in local minima.
optimizer = Adam(model.parameters(), lr=1e-4)

#-----Metrics------
dice_metric = DiceMetric(include_background=False, reduction="mean")
post_pred = AsDiscrete(threshold=0.5)  # Convert probabilities to binary predictions
post_label = AsDiscrete(threshold=0.5)   # Ensure labels are binary (0 or 1)

#-----Training loop------
num_epochs = 5
best_dice = 0.0

print(f"\nStarting training for {num_epochs} epochs...\n")

for epoch in range(num_epochs):
    # --- Training phase ---
    model.train()  # Set model to training mode
    epoch_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        # Load image and label to device
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Step 1: Zero gradients
        # Must clear before each backward pass
        # Otherwise gradients accumulate incorrectly
        optimizer.zero_grad()

        # Step 2: Forward pass
        predictions = model(images)

        # Step 3: Compute loss
        loss = loss_fn(predictions, labels)

        # Step 4: Backward pass
        # PyTorch calculates gradients for all weights
        loss.backward()

        # Step 5: Update weights
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    # --- Validation phase ---
    model.eval()  # Set model to evaluation mode
    dice_metric.reset()

    with torch.no_grad():  # No gradients needed for validation
        for val_batch in val_loader:
            val_images = val_batch["image"].to(device)
            val_labels = val_batch["label"].to(device)

            val_predictions = model(val_images)
            val_predictions = (val_predictions > 0.5).float()

            dice_metric(y_pred=val_predictions, y=val_labels)

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Loss: {avg_loss:.4f} | "
          f"Val Dice: {mean_dice:.4f}")

    # Save best model
    if mean_dice > best_dice:
        best_dice = mean_dice
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  New best model saved! Dice: {best_dice:.4f}")

print(f"\nTraining complete. Best Dice: {best_dice:.4f}")