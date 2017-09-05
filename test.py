import tensorflow as tf
import numpy as np
import os
import cv2
import argparse
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from sympy.geometry import *

class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = cfg.CARPLANE_CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print 'Restoring weights from: ' + self.weights_file
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def draw_result(self, img, result):
        for i in range(len(result)):
            x = np.zeros(4)
            y = np.zeros(4)
            for id in range(4):
                x[id] = int(result[i][2*id + 1])
                y[id] = int(result[i][2*id + 2])

            # x0 = int(result[i][1])
            # y0 = int(result[i][2])
            # x1 = int(result[i][3])
            # y1 = int(result[i][4])
            # x2 = int(result[i][5])
            # y2 = int(result[i][6])
            # x3 = int(result[i][7])
            # y3 = int(result[i][8])
            x = np.array(x)
            y = np.array(y)
            pts = np.stack((x, y), axis = 1)

            pts = pts.reshape((-1, 1, 2))
            print 'pts: ', pts
            cv2.polylines(img, np.int32([pts]), True, (0, 255, 255))
            #cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            #cv2.rectangle(img, (x - w, y - h - 20),
            #              (x + w, y - h), (125, 125, 125), -1)
            #cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.CV_AA)
            #cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX,
                      # 0.5, (0, 0, 0), 1)
    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]
        print 'shape', np.shape(result)
        print 'result', result
        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)
            result[i][5] *= (1.0 * img_w / self.image_size)
            result[i][6] *= (1.0 * img_h / self.image_size)
            result[i][7] *= (1.0 * img_w / self.image_size)
            result[i][8] *= (1.0 * img_h / self.image_size)
            print 'result: ', result[i]
        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        print 'shape', np.shape(net_output)
        print 'net_output', net_output
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))
        class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))
        scales = np.reshape(output[self.boundary1:self.boundary2], (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 8))
        print 'boxes0: ', boxes[1][1][0]

        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                         [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))

        print 'offset: ', offset
        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, 2] += offset
        boxes[:, :, :, 3] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, 4] += offset
        boxes[:, :, :, 5] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, 6] += offset
        boxes[:, :, :, 7] += np.transpose(offset, (1, 0, 2))
        print 'boxes offset: ', boxes[1][1][0]
        #boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        #boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size/self.cell_size

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        print 'shape: ', np.shape(filter_mat_probs)
        print 'filter_mat_probs: ', filter_mat_probs
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[
            0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:

                    ##TODO finishi the NMS
                    ##probs_filtered[j] = 0.0
                    pass
        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[
                          i][1], boxes_filtered[i][2], boxes_filtered[i][3], boxes_filtered[i][4], boxes_filtered[i][5],
                           boxes_filtered[i][6], boxes_filtered[i][7], probs_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def iou_poly(self, box1, box2):

        point = Point(box1[0], box1[1])
        print point
        p1, p2, p3, p4 = map(Point, [(box1[0], box1[1]), (box1[2], box1[3]), (box1[4], box1[5]), (box1[6], box1[7])])
        poly1 = Polygon(p1, p2, p3, p4)
        p5, p6, p7, p8 = map(Point, [(box2[0], box2[1]), (box2[2], box2[3]), (box2[4], box2[5]), (box2[6], box2[7])])
        poly2 = Polygon(p5, p6, p7, p8)
        poly1.intersection(poly2)

        intersect = poly1.intersection(poly2)
        if (len(intersect) == 3):
            interpoly = Polygon(intersect[0], intersect[1], intersect[2])
        elif (len(intersect) == 4):
            interpoly = Polygon(intersect[0], intersect[1], intersect[2], intersect[3])
        elif (len(intersect) == 5):
            interpoly = Polygon(intersect[0], intersect[1], intersect[2], intersect[3], intersect[4])
        elif (len(intersect) == 6):
            interpoly = Polygon(intersect[0], intersect[1], intersect[2], intersect[3], intersect[4], intersect[5])
        elif (len(intersect) == 7):
            interpoly = Polygon(intersect[0], intersect[1], intersect[2], intersect[3], intersect[4], intersect[5], intersect[6])
        elif (len(intersect) == 8):
            interpoly = Polygon(intersect[0], intersect[1], intersect[2], intersect[3], intersect[4], intersect[5], intersect[6], intersect[7])
        inter_square = interpoly.area
        union_square = tf.maximum(poly1.area + poly2.area - inter_square, 1e-10)
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)
    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):
        detect_timer = Timer()
        image = cv2.imread(imname)

        detect_timer.tic()
        result = self.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(detect_timer.average_time))

        self.draw_result(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="data/carplane/output/2017_09_05_13_41/save.ckpt-2000", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo = YOLONet(False)
    ##weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    weight_file = args.weights
    detector = Detector(yolo, weight_file)

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from image file
    imname = '/home/dingjian/data/carplane/images/P0188_plane.png'
    detector.image_detector(imname)


if __name__ == '__main__':
    main()
