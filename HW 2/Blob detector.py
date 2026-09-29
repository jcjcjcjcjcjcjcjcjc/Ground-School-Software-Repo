import cv2
from skimage import feature, color, io
import matplotlib.pyplot as plt

from pathlib import Path

image_path = Path(__file__).resolve().parent / "polka_dots_1.jpg"
image = color.rgb2gray(color.rgba2rgb(io.imread(str(image_path))))
 
blobs_log = feature.blob_log(image, max_sigma=30, num_sigma=10, threshold=0.1)
 
# Compute radii
blobs_log[:, 2] = blobs_log[:, 2] * (2 ** 0.5)
 
# Display
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
for y, x, r in blobs_log:
    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    ax.add_patch(c)
plt.show()
