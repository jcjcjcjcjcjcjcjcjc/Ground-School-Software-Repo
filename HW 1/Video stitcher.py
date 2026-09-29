# %%
import cv2
import matplotlib.pyplot as plt

video = cv2.VideoCapture("HW 1/Minecraft_stitch_test.mp4")
frames, i = [], 0

while True:
    ok, frame = video.read()
    if not ok:
        break
    if i%15 == 0:
        frames.append(cv2.resize(frame, None, fx=0.9, fy=0.9))
    i += 1

video.release()

stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
status, mosaic = stitcher.stitch(frames)
if status == cv2.Stitcher_OK:
    cv2.imwrite("HW 1/stiched_image.jpg", mosaic)
    plt.imshow(mosaic)
    plt.show()
else:
    print(status)