import cv2
import numpy as np


original_array = np.loadtxt("test.txt").reshape(300, 300, 4)[:, :, :-1]
cv2.imshow("image", original_array)

cv2.waitKey()
