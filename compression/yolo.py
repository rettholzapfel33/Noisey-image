## assuming in ./noisey-image/compression
## THIS IS A DEVELOPMENT FILE; DO NOT USE IN PRODUCTION
import sys
import os
sys.path.append('../')
from src import models
import cv2
from pathlib import Path
import numpy

def loadAndInferImage(image):
    yolo = models.YOLOv3(
        os.path.join('../src', 'obj_detector/cfg', 'coco.names'),
        os.path.join('../src', 'obj_detector/cfg', 'yolov3.cfg'),
        os.path.join('../src','obj_detector/weights','yolov3.weights')
    )
    yolo.initialize()
    
    '''
    box format:
    [ [x1, y1, x2, y2, conf, cls], [x1, y1, x2, y2, conf, cls], ... ]
                    ^                           ^
                   obj1                        obj2
    '''
    boxes = yolo.run(image) # format for one box: [x1, y1, x2, y2, confidence, class]

    return boxes

if __name__ == '__main__':
    image = cv2.imread('../imgs/default_imgs/car detection.png')
    assert type(image) != None, "file read fail"
    boxes = loadAndInferImage(image)
    boxes = boxes.numpy()

    for i in boxes:
        print(int(i[0]))
        print(int(i[1]))
        print(int(i[2]))
        print(int(i[3]))
