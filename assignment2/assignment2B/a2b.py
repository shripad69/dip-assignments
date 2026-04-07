from PIL import Image
import numpy as np

img = Image.open("input.jpeg")
r, g, b = img.split()

r_arr = np.array(r, dtype=np.float32)
g_arr = np.array(g, dtype=np.float32)
b_arr = np.array(b, dtype=np.float32)

factor = 1.5

r_enh = np.clip(r_arr * factor, 0, 255).astype(np.uint8)
g_enh = np.clip(g_arr * factor, 0, 255).astype(np.uint8)
b_enh = np.clip(b_arr * factor, 0, 255).astype(np.uint8)

enhanced = Image.merge('RGB', (
    Image.fromarray(r_enh),
    Image.fromarray(g_enh),
    Image.fromarray(b_enh)
))

enhanced.save("enhanced_image.jpg")