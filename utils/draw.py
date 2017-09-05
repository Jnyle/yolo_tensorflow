import cv2
import numpy as np
img = cv2.imread(r'/home/dingjian/code/yolo_tensorflow/data/carplane/trainval/images/P0001_car.png')
cv2.imshow('img', img)
cv2.waitKey()
print np.shape(img)