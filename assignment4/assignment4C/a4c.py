import cv2
import matplotlib.pyplot as plt

img = cv2.imread('input.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

R_c, G_c, B_c = cv2.split(contrast)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(contrast)
plt.title("Contrast Image")
plt.axis('off')

plt.subplot(2,2,2)
plt.hist(R_c.ravel(), bins=256, color='red')
plt.title("Red Histogram (Contrast)")

plt.subplot(2,2,3)
plt.hist(G_c.ravel(), bins=256, color='green')
plt.title("Green Histogram (Contrast)")

plt.subplot(2,2,4)
plt.hist(B_c.ravel(), bins=256, color='blue')
plt.title("Blue Histogram (Contrast)")

plt.tight_layout()
plt.savefig("output.png")