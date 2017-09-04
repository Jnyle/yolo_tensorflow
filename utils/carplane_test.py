from carplane import *
import yolo.config as cfg
import matplotlib.pylab as plt
import numpy as np
from skimage import draw, data

data = carplane('train')
batch1 = data.get()

img1 = batch1[0][0]
print np.shape(img1)

boxes = batch1[1][0]
print np.shape(boxes)


for y in range(7):
    for x in range(7):
        if (boxes[y][x][0] == True):
            box = np.array(boxes[y][x][1:9])
            x1 = np.array([box[0], box[2], box[4], box[6]], dtype=np.int64)
            y1 = np.array([box[1], box[3], box[5], box[7]], dtype=np.int64)
            x2 = x1
            y2 = 448 - y1
            rr1, cc1 = draw.line(y1[0], x1[0], y1[1], x1[1])
            rr2, cc2 = draw.line(y1[1], x1[1], y1[2], x1[2])
            rr3, cc3 = draw.line(y1[2], x1[2], y1[3], x1[3])
            rr4, cc4 = draw.line(y1[3], x1[3], y1[0], x1[0])

            draw.set_color(img1, [rr1, cc1], [1, 0, 0])
            draw.set_color(img1, [rr2, cc2], [1, 0, 0])
            draw.set_color(img1, [rr3, cc3], [1, 0, 0])
            draw.set_color(img1, [rr4, cc4], [1, 0, 0])
plt.imshow(img1)