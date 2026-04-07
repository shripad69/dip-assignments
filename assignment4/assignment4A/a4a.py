import cv2
import matplotlib.pyplot as plt

img = cv2.imread('input.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

R, G, B = cv2.split(img)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2,2,2)
plt.hist(R.ravel(), bins=256, color='red')
plt.title("Red Channel Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.subplot(2,2,3)
plt.hist(G.ravel(), bins=256, color='green')
plt.title("Green Channel Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.subplot(2,2,4)
plt.hist(B.ravel(), bins=256, color='blue')
plt.title("Blue Channel Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("output.png")