import torch
import math
import matplotlib.pyplot as plt
import imageio
import os

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")  # Uncomment this to run on GPU
#print(torch.cuda.is_available())
#print(torch.cuda.get_device_name(0))

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6

# Create a folder to store images
os.makedirs("images", exist_ok=True)

# Function to save images
def save_image(t, x, y, y_pred):
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), label="True")
    plt.plot(x.cpu().numpy(), y_pred.cpu().numpy(), label="Predicted")
    plt.legend()
    plt.title(f"Step {t}")
    plt.savefig(f"images/step_{t}.png")
    plt.clf()

# List to store image file paths
image_files = []

for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Save image for the current step if it's a multiple of 50
    if t % 50 == 0:
        save_image(t, x, y, y_pred)
        image_files.append(f"images/step_{t}.png")

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(
    f"Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3"
)

# Create gif
images = [imageio.imread(file) for file in image_files]
imageio.mimsave("training_animation.gif", images, duration=0.1)

# Remove image files
for file in image_files:
    os.remove(file)
