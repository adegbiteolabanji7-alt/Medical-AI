"""
convolutions.py
----------
convolutional Neural Networks(CNNs).

A regular neural network treats an image as a flat list of pixels.
It loses alll spactial information- it does not know that pixel(10,10)
is next to pixel (10,11).

A CNN preserves spatial structure. It slides a small filter across the image, detecting features- edges, textures, shapes - regardless of where the appear. This is why CNNs dominate medical imaging.

As a sonographer, when you look for a fetal headc, you are not looking at random pixels. You are looking for a specific shape  in a specific shape in a spatial relationship with surrounding structures. CNNs learn to do exactly this
"""

import torch
import torch.nn as nn

# A single greyscale medical image
#Shape: (batch,channel, height,width)
image = torch.randn(1, 1, 256, 256)
print(f" Input image shape: {image.shape}")

# A convolutional layer 
# in_channels = 1: greyscale input
# out_channels = 8: learn 8 different features
# kernel_size = 3: 3*3 filter sliding across image
#padding= 1: keeps input same size as output
conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
output = conv(image)
print(f"After conv2d: {output.shape}")
print("8 feature maps - each detecting a diffferent pattern")

#Pooling: reduce spatial size, keep impoertant features
pool = nn.MaxPool2d(kernel_size=2)
pooled = pool(output)
print(f"After MaxPool2d: {pooled.shape}")
print("Spatial size halved, less computation, key features kept")

# A proper CNN for medical image classification
class MedicalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size= 3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(262144, 64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x
model = MedicalCNN()
output=model(image)
print(f"\nCNN output shape: {output.shape}")
print(f"CNN prediction (probability): {output.item(): .4f}")
print("\nModel architecture:")
print(model)




