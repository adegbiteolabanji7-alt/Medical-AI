"""
first_network.py
-------------
Building and training a neural network from scratch.

we will train a network to classify whether a patients measuremeny is above or below a clinical threshhold. Simple problem, real pattern.

This is the full training loop, the same loop that trains segmentation models, just scaled up.

"""

import torch 
import torch.nn as nn

#___Data___
#Simulated clinical measurements(e.g normalised BI-RADS score)
# 0 = below threshold, 1 = above threshold.
torch.manual_seed(42)
x = torch.randn(100,3)
y = (x.sum(dim=1)>0).float().unsqueeze(1)
print(f"Input shape: {x.shape}")
print(f"Positive cases: {y.sum(). int()}/100")

#___Model___
class ClinicalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network= nn.Sequential(
            nn.Linear(3,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.network(x)
model = ClinicalClassifier()
print(f"\nModel architecture:\n{model}")

# ---Training setup----
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#---Training loop-----
print("\nTraining.....")
for epoch in range(100):
    # Forward pass: make prediction
    y_pred = model(x)

# Calculate loss: how wrong are we?
    loss = loss_fn(y_pred, y)
    
    # Backward pass: calculate gradients
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: Loss {loss.item():.4f}")
# ----Evaluation----
with torch.no_grad():
    predictions = (model(x) > 0.5).float()
    accuracy = (predictions == y).float().mean()
    print(f"\nFinal accuracy: {accuracy: .2%}")