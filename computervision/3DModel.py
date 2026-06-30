from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = Image.open("image2.jpg").convert("L")  # Convert to grayscale

# Resize for faster plotting
image = image.resize((200, 200))

# Convert to numpy array
z = np.array(image)

# Create X and Y coordinates
x = np.arange(z.shape[1])
y = np.arange(z.shape[0])
X, Y = np.meshgrid(x, y)

# Plot 3D surface
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, -z, cmap='gray')

ax.set_title("Simple 3D Surface from Image")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Height")

plt.show()
