import cv2
import matplotlib.pyplot as plt

img = cv2.imread("HW 2/polka_dots_3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
p = cv2.SimpleBlobDetector_Params()

p.filterByArea = True
p.minArea, p.maxArea = 30, 2500

p.filterByColor = True
p.blobColor = 0

p.filterByCircularity = True
p.minCircularity = 0.7

keypoints = cv2.SimpleBlobDetector_create(p).detect(gray)
result = cv2.drawKeypoints(img, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(result)
plt.show()
cv2.imwrite("HW 2/result 3.png", result)
print("polka_dots_3.png", len(keypoints), "dots")