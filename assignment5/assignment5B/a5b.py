import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

Q_base = np.array([
[16,11,10,16,24,40,51,61],
[12,12,14,19,26,58,60,55],
[14,13,16,24,40,57,69,56],
[14,17,22,29,51,87,80,62],
[18,22,37,56,68,109,103,77],
[24,35,55,64,81,104,113,92],
[49,64,78,87,103,121,120,101],
[72,92,95,98,112,100,103,99]
])

img = cv2.imread('input.jpeg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

h, w = img.shape

def compress(Qf):
    Q = Q_base * Qf
    out = np.zeros_like(img, dtype=np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = img[i:i+8, j:j+8]
            d = dct2(block)
            q = np.round(d / Q)
            deq = q * Q
            r = idct2(deq)
            out[i:i+8, j:j+8] = r

    return np.clip(out, 0, 255).astype(np.uint8)

Q_values = [5, 15, 30]
results = [compress(Q) for Q in Q_values]

fig, axs = plt.subplots(1, 4, figsize=(18, 5))

axs[0].imshow(img, cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis('off')

for i, Q in enumerate(Q_values):
    axs[i+1].imshow(results[i], cmap='gray')
    axs[i+1].set_title(f"Q = {Q}")
    axs[i+1].axis('off')

plt.suptitle("Quantization Effect on Image Quality")
plt.tight_layout()
plt.savefig("output.png")