from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# STEP 1: Load grayscale image
# -----------------------------
img = Image.open("input.jpeg").convert('L')
gray = np.array(img)

# -----------------------------
# STEP 2: Bit-plane reconstruction
# -----------------------------
recon_8bit = gray
recon_4bit = (gray >> 4) << 4
recon_1bit = (gray & 1) * 255

# Save images
Image.fromarray(recon_8bit).save("recon_8bit.png")
Image.fromarray(recon_4bit.astype(np.uint8)).save("recon_4bit.png")
Image.fromarray(recon_1bit.astype(np.uint8)).save("recon_1bit.png")

# -----------------------------
# STEP 3: Compute PSNR + Size
# -----------------------------
orig = gray.astype(float)

psnr_vals = []
sizes = []
names = ["recon_8bit.png", "recon_4bit.png", "recon_1bit.png"]

for name in names:
    rec = np.array(Image.open(name).convert('L'), dtype=float)

    mse = np.mean((orig - rec) ** 2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))

    psnr_vals.append(psnr)
    sizes.append(os.path.getsize(name))

print("PSNR values (dB):", psnr_vals)
print("File sizes (bytes):", sizes)

# -----------------------------
# STEP 4: Plot graphs
# -----------------------------
x_labels = ['8 bits', '4 bits', '1 bit']
x = [1, 2, 3]

plt.figure(figsize=(10, 4))

# PSNR graph
plt.subplot(1, 2, 1)
plt.plot(x, psnr_vals, 'o-')
plt.xticks(x, x_labels)
plt.ylabel('PSNR (dB)')
plt.title('PSNR vs Bits Used')

# File size graph
plt.subplot(1, 2, 2)
plt.plot(x, sizes, 's-')
plt.xticks(x, x_labels)
plt.ylabel('File Size (bytes)')
plt.title('File Size vs Bits Used')

plt.tight_layout()
plt.savefig("results.png")
plt.show()