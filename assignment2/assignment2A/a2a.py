import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = np.array(Image.open('input.jpeg').convert('RGB'))

R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(R, cmap='Reds', vmin=0, vmax=255)
plt.title('Red Channel')

plt.subplot(1, 3, 2)
plt.imshow(G, cmap='Greens', vmin=0, vmax=255)
plt.title('Green Channel')

plt.subplot(1, 3, 3)
plt.imshow(B, cmap='Blues', vmin=0, vmax=255)
plt.title('Blue Channel')

plt.tight_layout()
plt.savefig("output.png")