import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from utils import *
from utils import _draw_and_return_output_image
import cv2

def load_jit_network(weight_path):
    model = torch.jit.load(weight_path)
    return model

def detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5):
    """Inferences one image with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
    :rtype: nd.array
    """

    DEFAULT_TRANSFORMS = transforms.Compose([
        AbsoluteLabels(),
        PadSquare(),
        RelativeLabels(),
        ToTensor(),
    ])

    model.eval()  # Set model to evaluation mode

    # Configure input
    '''
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])((image, np.zeros((1, 5))))[0].unsqueeze(0)
    '''
    preproc = transforms.Compose([transforms.ToTensor(), Resize(img_size)])
    input_img = preproc(image)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    #return to_cpu(detections).numpy()
    #return to_cpu(detections)
    return detections.detach().cpu()

def run_frame(np_arr):
    return 0

if __name__ == '__main__':
    print(colorstr('bright_magenta', 'bold', "Debugging purposes ONLY"))
    
    print("0) Load metadata...", end=' ')
    CLASSES = load_classes('../darknet_radar_cfg/radar.names')
    print(colorstr('green', 'Done!\n'))

    print("1) Load model...", end=' ')
    #model = load_jit_network('./yolov3_coco_jit.pt')
    model = load_jit_network('yolov3_syn_walk_radar_jit.pt')
    print(colorstr('green', 'Done!\n'))
    
    print("2) Detecting image...")
    #input_image = cv2.imread('../dog.png')
    input_image = cv2.imread('/home/vijay/Documents/devmk4/radar-cnn/data/syn_walk/images/frame_40_40.png')
    assert input_image is not None, "Image was not read properly"
    detections = detect_image(model, input_image)
    print(colorstr('green', 'Done!\n'))
    print(colorstr('green', 'bold', "%i detections:"%(len(detections))))
    for det in detections:
        print("\t+ class: %s | confidence: %0.3f | bbox: [%i, %i, %i, %i] "%( CLASSES[det[-1].int().item()], det[-2], *(det[:-2].int()) ))
    
    output_path = 'out.png'
    print("\n3) Drawing detections out to: %s"%(output_path), end=' ')
    out_img = _draw_and_return_output_image(input_image, detections, 416, CLASSES)
    cv2.imwrite(output_path, out_img)
    print(colorstr('green', 'Done!\n'))
