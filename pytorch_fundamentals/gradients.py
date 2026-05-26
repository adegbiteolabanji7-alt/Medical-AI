"""

gradients.py
-------------
How neural networks learn: gradients and backpropagation.

A neural network learns by making a prediction, measuring how wrong it was (the loss), then adjusting its weights to be less wrong next time. gradients tell it which direction to adjust and by howv much.

This is called back propagation, pyTorch does it automatically.
"""

import torch

# requires_grad = True tells pyTorch to track this sensor.
# for gradient computation. Think of it as saying:
# "this is a weight i want to learn"
w = torch.tensor(2.0,requires_grad = True)

# A simple prediction: y = w * x
x = torch.tensor(3.0)
y_pred = w * x 
print (f"Weight w: {w}")
print(f"Input x: {x}")
print(f"Prediction y_pred: {y_pred}")

# The true value we wanted 
y_true = torch.tensor(9.0)

#Loss: how wrong were we?
# Mean squared error: (prediction - truth)^2
loss = (y_pred - y_true) ** 2
print(f"True value: {y_true}")
print(f"Loss: {loss}")

#Backpropagation: Pytorch calcu;ates the gradient automatically
# This answers: if i increase w slightly, does the loss go up or down?
loss.backward()

# The gradient tells us the direction and magnitude to adjust w
print(f"\nGradient of loss with respect to w: {w.grad}")
print("This tells us: increase w to reduce the loss")

# Gradient descent: adjust w in the opposite direction of the gradient
learning_rate = 0.01
with torch.no_grad():
    w -= learning_rate * w.grad

print(f"\nUpdated weight w: {w}")
print("w moved closer to the value that minimises loss")
