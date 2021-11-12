import torch
import sys
import os
import numpy as np
sys.path.append('../../')
from src.obj_detector.detect import detect, load_model, detect_image, _draw_and_save_output_image, _draw_and_return_output_image
from src.obj_detector.utils.utils import load_classes
import cv2

dummy_input = torch.Tensor(1,3,416,416)
dummy_input = dummy_input.cuda()

'''
_model = load_model(
    #os.path.join('src', 'obj_detector/cfg', 'coco.names'),
    os.path.join('../', 'obj_detector/cfg', 'yolov3.cfg'),
    os.path.join('../','obj_detector/weights','yolov3.weights')
)
'''

_model = load_model(
    #os.path.join('src', 'obj_detector/cfg', 'coco.names'),
    os.path.join('./darknet_radar_cfg', 'yolov3_radar_5.cfg'),
    os.path.join('./darknet_radar_cfg','yolov3_radar_5_final.weights')
)

CLASSES = load_classes('./darknet_radar_cfg/radar.names')
#CLASSES = load_classes('../obj_detector/cfg/coco.names')

_model.eval()
_ = _model(dummy_input)

traced_module = torch.jit.trace(_model, dummy_input)
#traced_module.cpu()
#out = traced_module(dummy_input)

#img = np.ndarray((416,416,3)).astype(np.uint8)
#img = cv2.imread('/home/vijay/Documents/devmk4/yolov4/darknet/data/dog.jpg')
img = cv2.imread('/home/vijay/Documents/devmk4/radar-cnn/data/syn_walk/images/frame_40_40.png')
#img = cv2.imread('/home/vijay/Documents/devmk4/Noisey-image/imgs/default_imgs/car detection.png')
#out = detect_image(_model, img)

# Test traced model:
out = detect_image(traced_module, img)
print(out)
out_img = _draw_and_return_output_image(img, out, 416, CLASSES)
cv2.imwrite('out.png', out_img)
#cv2.imshow('test', out_img)
#cv2.waitKey(-1)

traced_module.save('./yolo_test/yolov3_syn_walk_radar_jit.pt')
#traced_module.save('./yolo_test/yolov3_coco_jit.pt')