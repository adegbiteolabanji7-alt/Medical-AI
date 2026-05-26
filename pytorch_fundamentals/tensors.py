"""
tensors.py
-------------
PyTorch fundamentals: Tensors.

A tensor is the core data structure in pyTorch.
Everything - images, model weights, predictions - is a tensor.

If youve used numpy arrays, tensors are the same idea but with two superpowers:
1. They can run on GPU
2. They track gradients for automatic differentiation (how neural networks learn)
"""

import torch

# A 1D tensor - like a list of numbers
a = torch.tensor([1.0, 2.0, 3.0])
print(f"1D tensor: {a}")
print(f"Shape: {a.shape}")
print(f"Data type: {a.dtype}")
      
# A 2D tensor- like a matrix or a grayscale image
b = torch.tensor([[1.0,2.0],[3.0,4.0]])
print(f"n2D tensor : \n{b}")
print(f"Shape: {b.shape}")

# A 3D tensor - like a batch of grayscale images
# Shape: (batch_size, height,width)
c = torch.zeros(4,128,128)
print(f"\n3D tensor shape (4 images, 128*128): {c.shape}")

# 4D tensor - like a batch of RGB images
# Shape: (batch_size, channels, height, width)
d = torch.zeros(4, 3, 128, 128)
print (f"4D tensor shape ( 4 RGB images, 128 * 128): {d.shape}")

# Why does this matter for medical imaging?
# A CT scan volume is a 3D tensor: (depth, height, width)
# A batch of CT scans is 4D: (batch, depth, height, width)
#MONAI adds a channel dimension: (batch, channel, depth, height,width)
ct_scan = torch.zeros (1, 1, 64,512,512)
print(f"\nCT scan tensor shape: {ct_scan.shape}")
print("(batch=1, channel=1, depth=46, height=512, width=512)")


# Tensor operations — the maths behind neural networks
x = torch.tensor([2.0, 3.0, 4.0])
y = torch.tensor([1.0, 1.0, 1.0])

print(f"\nAddition: {x + y}")
print(f"Multiplication: {x * y}")
print(f"Mean: {x.mean()}")
print(f"Sum: {x.sum()}")

# Reshape — critical skill
# Same data, different shape
a = torch.zeros(4, 128, 128)
b = a.reshape(4, -1)
print(f"\nOriginal shape: {a.shape}")
print(f"Reshaped: {b.shape}")
print("(-1 means: figure out this dimension automatically)")