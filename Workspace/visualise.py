# requirements: numpy, opencv-python
import numpy as np
import cv2

from sys import argv

lines = open(argv[1]).readlines()

for line in lines[1:]:
    numbers = np.array([float(v) for v in line.split(' ')[1:-1]]).reshape(16, 16)
    cv2.imshow("image", numbers)
    cv2.waitKey(0)
    break