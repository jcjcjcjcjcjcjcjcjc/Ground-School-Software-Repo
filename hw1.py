# %%
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("photo.jpg")
if img is None:
    raise FileNotFoundError("Could not find photo.jpg")

edges = cv2.Canny(img, 100, 200)

plt.imshow(edges, cmap="gray")
plt.axis("off")
plt.savefig("edges.png")
plt.show()