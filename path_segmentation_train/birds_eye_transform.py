from pathlib import Path
import cv2
import numpy as np
import time

# We have to transform a sample image first to test whether the bird's eye view transformation is correct
# This is a proof of concept test script
# Note that the resolution of input images are 320x180

DELAY = 250  # milliseconds

# Read a sample image
images_path = Path(__file__).parent.parent / "dataset" / "yolo" / "images"

# Select four points in the original image that form a trapezoid
tl = (131, 105)  # top-left
bl = (0, 127)    # bottom-left
tr = (189, 105)  # top-right
br = (320, 127)  # bottom-right
src_points = np.array([tl, bl, tr, br], dtype=np.float32)
# Define the destination points for the bird's eye view (a rectangle)
dest_points = np.array([[0, 0], [0, 180], [320, 0], [320, 180]], dtype=np.float32)
matrix = cv2.getPerspectiveTransform(src_points, dest_points)

for file in sorted(images_path.glob("*.png")):
    print(f"Processing file: {file.name}")
    image = cv2.imread(str(file))

    cv2.circle(image, tl, 1, (0,0,255), -1)
    cv2.circle(image, bl, 1, (255,0,0), -1)
    cv2.circle(image, tr, 1, (0,255,0), -1)
    cv2.circle(image, br, 1, (0,255,255), -1)
    bird_eye_view = cv2.warpPerspective(image, matrix, (320, 180))

    cv2.imshow("Original Image with Points", image)
    cv2.imshow("Bird's Eye View", bird_eye_view)

    if cv2.waitKey(DELAY) == 27: # ESC key to exit
        break

# cv2.destroyAllWindows()
