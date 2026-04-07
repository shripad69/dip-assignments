import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
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

def compute_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def compress(img, Qf):
    h, w = img.shape
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

original = cv2.imread('input.jpeg', cv2.IMREAD_GRAYSCALE)
original = cv2.resize(original, (256, 256))

Q_values = [5, 15, 30]

psnr_values = []
compression_ratios = []
images = []

for Q in Q_values:
    comp = compress(original, Q)
    images.append(comp)

    filename = f"output_Q{Q}.jpg"
    cv2.imwrite(filename, comp)

    psnr = compute_psnr(original, comp)
    psnr_values.append(psnr)

    orig_size = os.path.getsize('input.jpeg')
    comp_size = os.path.getsize(filename)

    ratio = orig_size / comp_size
    compression_ratios.append(ratio)

    print(f"Q={Q} | PSNR={psnr:.2f} dB | CR={ratio:.2f}")

fig1, axs = plt.subplots(1, 4, figsize=(18, 5))

axs[0].imshow(original, cmap='gray')
axs[0].set_title("Original")
axs[0].axis('off')

for i, Q in enumerate(Q_values):
    axs[i+1].imshow(images[i], cmap='gray')
    axs[i+1].set_title(f"Q={Q}\nPSNR={psnr_values[i]:.2f}\nCR={compression_ratios[i]:.2f}")
    axs[i+1].axis('off')

plt.tight_layout()
plt.savefig("output.png")

plt.figure()
plt.plot(Q_values, psnr_values, marker='o')
plt.xlabel("Quantization Factor (Q)")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs Quantization Factor")
plt.grid()
plt.savefig("output.png")

plt.figure()
plt.plot(Q_values, compression_ratios, marker='o')
plt.xlabel("Quantization Factor (Q)")
plt.ylabel("Compression Ratio")
plt.title("Compression Ratio vs Quantization Factor")
plt.grid()
plt.savefig("output.png")