import cv2
import numpy as np

original = cv2.imread("../imgs/car detection.png")
duplicate = cv2.imread("../imgs/car_detection_0noise.png")

if original.shape == duplicate.shape:
    print("The images have same sizes and channels")
else:
    print("The images have different sizes or channels")
    exit()

difference = cv2.subtract(original, duplicate)
b, g, r = cv2.split(difference)

if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
    print("The images are completely Equal")
else:
    print("But the images are not equal")