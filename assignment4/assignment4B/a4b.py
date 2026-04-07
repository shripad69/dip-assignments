import cv2
import matplotlib.pyplot as plt

img = cv2.imread('input.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

bright = cv2.convertScaleAbs(img, alpha=1, beta=50)

R_b, G_b, B_b = cv2.split(bright)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(bright)
plt.title("Bright Image")
plt.axis('off')

plt.subplot(2,2,2)
plt.hist(R_b.ravel(), bins=256, color='red')
plt.title("Red Histogram (Bright)")

plt.subplot(2,2,3)
plt.hist(G_b.ravel(), bins=256, color='green')
plt.title("Green Histogram (Bright)")

plt.subplot(2,2,4)
plt.hist(B_b.ravel(), bins=256, color='blue')
plt.title("Blue Histogram (Bright)")

plt.tight_layout()
plt.savefig("output.png")