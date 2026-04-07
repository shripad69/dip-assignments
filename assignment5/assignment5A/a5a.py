import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

img = cv2.imread('input.jpeg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

row, col = 80, 80
block = img[row:row+8, col:col+8]

dct_block = dct2(block)
dct_log = np.log(abs(dct_block) + 1)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(img, cmap='gray')
axs[0].add_patch(plt.Rectangle((col, row), 8, 8,
edgecolor='red', facecolor='none', linewidth=2))
axs[0].set_title("Grayscale Image\n(selected block in red)")
axs[0].axis('off')

axs[1].imshow(block, cmap='gray')
for i in range(8):
    for j in range(8):
        axs[1].text(j, i, str(block[i, j]),
        ha='center', va='center', color='yellow', fontsize=8)
axs[1].set_title(f"8x8 Pixel Block\n[row={row}, col={col}]")
axs[1].axis('off')

im = axs[2].imshow(dct_log, cmap='inferno')
for i in range(8):
    for j in range(8):
        axs[2].text(j, i, f"{dct_block[i,j]:.0f}",
        ha='center', va='center', color='white', fontsize=7)

axs[2].set_title("2-D DCT Coefficients\n(log scale | DC = top-left)")
axs[2].axis('off')

fig.colorbar(im, ax=axs[2])
plt.tight_layout()
plt.savefig("output.png")