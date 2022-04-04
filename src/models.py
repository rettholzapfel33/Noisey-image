import abc
import os
from pathlib import Path
from pyexpat import model
from PyQt5.QtCore import QObject
import os, csv, torch, scipy.io
import numpy as np

from src.predict_img import new_visualize_result, process_img, predict_img, load_model_from_cfg, visualize_result, transparent_overlays, get_color_palette
from src.mit_semseg.utils import AverageMeter, accuracy, intersectionAndUnion

# import yolov3 stuff:
import src.obj_detector.detect as detect
from src.obj_detector.models import load_model
from src.obj_detector.utils.utils import load_classes

# import efficientNetV2
import json
from src.efficientnet_pytorch.model import EfficientNet
from torchvision import transforms

# import yolov4 stuff:
from src.yolov4.tool.utils import *
from src.yolov4.tool.darknet2pytorch import Darknet

currPath = str(Path(__file__).parent.absolute()) + '/'

class Model(abc.ABC):
    """
    Creates and adds models. 
    Requirment: The network needs to be fitted in four main funtions: run, initialize, deinitialize, and draw.   
    """
    def __init__(self, *network_config) -> None:
        self.complexOutput = False
    
    @abc.abstractclassmethod
    def run(self, input):
        raise NotImplementedError

    @abc.abstractclassmethod
    def initialize(self, *kwargs):
        raise NotImplementedError

    @abc.abstractclassmethod
    def deinitialize(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def draw(self, pred):
        raise NotImplementedError

    @abc.abstractclassmethod
    def draw_single_class(self, pred):
        raise NotImplementedError

    @abc.abstractclassmethod
    def report_accuracy(self):
        raise NotImplementedError

    @abc.abstractproperty
    def outputFormat(self):
        raise NotImplementedError


    def __call__(self):
        pred = self.run()
        return pred

class Segmentation(Model):
    """
    Segmentation Model that inherits the Model class
    It specifies its four main functions: run, initialize, deinitialize, and draw. 
    """
    def __init__(self, *network_config) -> None:
        super().__init__(network_config)
        
        self.complexOutput = True
        self.cfg, self.colors = network_config
        #self.cfg = str(Path(__file__).parent.absolute()) + "/config/ade20k-hrnetv2.yaml"
        # colors
        #self.colors = scipy.io.loadmat(str(Path(__file__).parent.absolute()) + '/data/color150.mat')['colors']
        self.names = {}
        self.complexOutput = True # output is a large matrix. Saving output is a little different than object detector

        with open(str(Path(__file__).parent.absolute()) + '/data/object150_info.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.names[int(row[0])] = row[5].split(";")[0]

    def run(self, input):
        if torch.cuda.is_available():
            self.segmentation_module.cuda()
        else:
            self.segmentation_module.cpu()

        img_original, singleton_batch, output_size = process_img(frame = input)

        try:
            # predict
            img_original, singleton_batch, output_size = process_img(frame = input)
            pred = predict_img(self.segmentation_module, singleton_batch, output_size)
        except:
            self.segmentation_module.cpu()

            print("Using cpu")

            # predict
            img_original, singleton_batch, output_size = process_img(frame = input, cpu = 1)
            pred = predict_img(self.segmentation_module, singleton_batch, output_size)
        return pred

    def initialize(self, *kwargs):
        # Network Builders
        print("parsing {}".format(self.cfg))
        self.segmentation_module = load_model_from_cfg(self.cfg)
        
        self.segmentation_module.eval()
        return 0

    def deinitialize(self):
        return -1

    def draw(self, pred, img):
        detectedNames = {"all": [255,255,255]}
        pred_color, org_pred_split = visualize_result(img, pred, self.colors)

        color_palette = get_color_palette(pred, org_pred_split.shape[0], self.names, self.colors, detectedNames)

        # transparent pred on org
        dst = transparent_overlays(img, pred_color, alpha=0.6)

        return {"dst": dst, 
        "segmentation": pred_color, 
        "listOfNames":detectedNames
                }

    def draw_single_class(self, pred, img, selected_class):
        imgs = new_visualize_result(pred, img, selected_class)
        return {"segmentation": imgs[0], "overlay": imgs[1]}

    def report_accuracy(self, pred, pred_truth):
        acc_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()

        acc, pix = accuracy(pred, pred_truth)
        intersection, union = intersectionAndUnion(pred, pred_truth, 150)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        
        class_ious = {}
        iou = intersection_meter.sum / (union_meter.sum + 1e-10)
        for i, _iou in enumerate(iou):
            class_ious[i] = _iou
        return iou.mean(), acc_meter.average(), class_ious

    def outputFormat(self):
        return "{}" # hex based output?

    def calculateRatios(self, dets):
        values, counts = np.unique(dets, return_counts=True)
        total_idx = [i for i in range(150)]
        for idx in total_idx:
            if not idx in values:
                counts = np.insert(counts, idx, 0)
        return counts

class YOLOv3(Model):
    """
    YOLO Model that inherits the Model class
    It specifies its four main functions: run, initialize, deinitialize, and draw. 
    """
    def __init__(self, *network_config) -> None:
        super(YOLOv3, self).__init__()
        # network_config: CLASSES, CFG, WEIGHTS
        self.CLASSES, self.CFG, self.WEIGHTS = network_config
        # self.CLASSES = os.path.join(currPath, 'obj_detector/cfg', 'coco.names')
        # self.CFG = os.path.join(currPath, 'obj_detector/cfg', 'yolov3.cfg')
        # self.WEIGHTS = os.path.join(currPath,'obj_detector/weights','yolov3.weights')
        print(self.CLASSES, self.CFG, self.WEIGHTS)
        self.classes = load_classes(self.CLASSES)

    def run(self, input):
        pred = detect.detect_image(self.yolo, input)
        return pred #[x1,y1,x2,y2,conf,class] <--- box

    def initialize(self, *kwargs):
        self.yolo = load_model(self.CFG, self.WEIGHTS)
        return 0
    
    def deinitialize(self):
        return -1
    
    def draw(self, pred, img):
        np_img, detectedNames = detect._draw_and_return_output_image(img, pred, 416, self.classes)
        return {"dst": np_img,
                "listOfNames":detectedNames}

    def draw_single_class(self, pred, img, selected_class):
        np_img = detect._draw_and_return_output_image_single_class(img, pred, selected_class, self.classes)
        return {"overlay": np_img}

    def report_accuracy(self, pred, pred_truth):
        print('pred comparison', pred, pred_truth)
        return

    def outputFormat(self):
        return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"

class EfficientNetV2(Model):
    def __init__(self, *network_config) -> None:
        super(EfficientNetV2, self).__init__()

        self.cfg = network_config

    def initialize(self, *kwargs):
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model.eval()

        # Load ImageNet class names for returning predictions
        self.labels_map = json.load(open(os.path.join('src', 'efficientnet_pytorch', 'labels_map.txt')))
        self.labels_map = [self.labels_map[str(i)] for i in range(1000)]

    def run(self, input):
        tfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        img = tfms(input).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(img)
        print('~~~~~~~~~OUTPUT~~~~~~~~~~~: ', output.shape)
        for idx in torch.topk(output, k=5).indices.squeeze(0).tolist():
            prob = torch.softmax(output, dim=1)[0, idx].item()
            print('{label:<75} ({p:.2f}%)'.format(label=self.labels_map[idx], p=prob*100))
        return output

    def deinitialize(self):
        return -1

    def draw(self, pred, img):
        print(pred)

    def draw_single_class(self, pred, img, selected_class):
        # np_img = detect._draw_and_return_output_image_single_class(img, pred, selected_class, self.classes)
        # return {"overlay": np_img}
        return

    def report_accuracy(self, pred, pred_truth):
        print(pred, pred_truth)
        # print(self.model.summary())
        acc_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()

        acc, pix = accuracy(pred, pred_truth)
        intersection, union = intersectionAndUnion(pred, pred_truth, 150)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        
        class_ious = {}
        iou = intersection_meter.sum / (union_meter.sum + 1e-10)
        for i, _iou in enumerate(iou):
            class_ious[i] = _iou
        return iou.mean(), acc_meter.average(), class_ious

    def outputFormat(self):
        return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"

class DETR(Model):
    """
    DETR Model that inherits the Model class
    It specifies its four main functions: run, initialize, deinitialize, and draw.
    """
    def __init__(self, *network_config) -> None:
        super(DETR, self).__init__()
        self.CLASSES = network_config[0]
        print(self.CLASSES)
        self.classes = load_classes(self.CLASSES)
    
    def initialize(self, *kwargs):
        self.detr = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
        self.detr= self.detr.eval()
        if torch.cuda.is_available():
            self.on_gpu = True
            self.detr.cuda()
        else:
            self.on_gpu = False
            self.detr.cpu()
        return 0

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        # Push to CPU to perform operation after
        b = self.box_cxcywh_to_xyxy(out_bbox).cpu()
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def run(self, input):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(input).unsqueeze(0)
        if self.on_gpu: img = img.cuda()
        pred = self.detr(img)
        return pred

    def deinitialize(self):
        # deinitialize equivalent of detr model if there is one
        return -1

    def draw(self, pred, img):
        np_img, detectedNames = img, 1 #TODO: detect._draw_and_return_output_image equivalent from yolo equivalent in detr
        return {"dst": np_img,
                "listOfNames": detectedNames}

    def draw_single_class(self, pred, img, selected_class):
        np_img = img #TODO: detect._draw_and_return_output_image_single_class from yolo equivalent in detr
        return {"overlay": np_img}

    def report_accuracy(self, pred, pred_truth):
        return

    def outputFormat(self):
        return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"

class YOLOv4(Model):
    """
    YOLOv4 Model that inherits the Model class
    It specifies its four main functions: run, initialize, deinitialize, and draw. 
    """
    def __init__(self, *network_config) -> None:
        super(YOLOv4, self).__init__()
        # network_config: CLASSES, CFG, WEIGHTS
        self.CLASSES, self.CFG, self.WEIGHTS = network_config
        
        print(self.CLASSES, self.CFG, self.WEIGHTS)
        self.classes = load_class_names(self.CLASSES)

    def run(self, input):
        #pred = detect.detect_image(self.yolo, input)
        #return pred #[x1,y1,x2,y2,conf,class] <--- box
        return

    def initialize(self, *kwargs):
        self.yolo = Darknet(self.CFG)
        self.yolo.load_weights(self.WEIGHTS)
        #self.yolo = load_model(self.CFG, self.WEIGHTS)
        # need to use tools/darknet2pytorch.py to convert weights -- is that already done there?
        return 0
    
    def deinitialize(self):
        return -1
    
    def draw(self, pred, img):
        np_img, detectedNames = detect._draw_and_return_output_image(img, pred, 416, self.classes)
        return {"dst": np_img,
                "listOfNames":detectedNames}

    def draw_single_class(self, pred, img, selected_class):
        np_img = detect._draw_and_return_output_image_single_class(img, pred, selected_class, self.classes)
        return {"overlay": np_img}

    def report_accuracy(self, pred, pred_truth):
        print('pred comparison', pred, pred_truth)
        return

    def outputFormat(self):
        return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"


_registry = {
    'Semantic Segmentation': Segmentation(
        str(Path(__file__).parent.absolute()) + "/mit_semseg/config/ade20k-hrnetv2.yaml",
        scipy.io.loadmat(str(Path(__file__).parent.absolute()) + '/data/color150.mat')['colors']
    ),
    'Object Detection (YOLOv3)': YOLOv3(
        os.path.join(currPath, 'obj_detector/cfg', 'coco.names'),
        os.path.join(currPath, 'obj_detector/cfg', 'yolov3.cfg'),
        os.path.join(currPath,'obj_detector/weights','yolov3.weights')
    ),
    'EfficientNetV2': EfficientNetV2(
        'efficientnet-b0'
    ),
    'Object Detection (DETR)': DETR(
        os.path.join(currPath, 'obj_detector/cfg', 'coco.names')
    ),
    'Object Detection (YOLOv4)': YOLOv4(
        os.path.join(currPath, 'yolov4/data', 'coco.names'),
        os.path.join(currPath, 'yolov4/cfg', 'yolov4.cfg'),
        os.path.join(currPath,'yolov4/weight','yolov4.weights')
    )
}