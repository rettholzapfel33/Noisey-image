import torch
import os
import sys
import cv2
import numpy as np
import detect
from models import load_model
from utils.utils import load_classes, rescale_boxes, non_max_suppression, to_cpu, print_environment_info


if __name__ == '__main__':
    # load in weights:
    CLASSES = '/home/vijay/Documents/devmk4/yolov4/darknet/data/coco.names'
    CFG = '/home/vijay/Documents/devmk4/yolov4/darknet/cfg/yolov3.cfg'
    WEIGHTS = '/home/vijay/Documents/devmk4/yolov4/darknet/yolov3.weights'
    
    yolo = load_model(CFG, WEIGHTS)
    classes = load_classes(CLASSES)  # List of class names
    print(yolo)

    # Load in image:
    img = cv2.imread('/home/vijay/Documents/devmk4/yolov4/darknet/data/dog.jpg')
    dets = detect.detect_image(yolo, img)
    out_img = detect._draw_and_return_output_image(img, dets, 416, classes)
    cv2.imwrite('test.png', out_img)
    #cv2.imshow('test', out_img)
    #cv2.waitKey(-1)
