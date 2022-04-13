import abc
import os
from pathlib import Path
from PyQt5.QtCore import QObject
import os, csv, torch, scipy.io
import numpy as np

from src.predict_img import new_visualize_result, process_img, predict_img, load_model_from_cfg, visualize_result, transparent_overlays, get_color_palette
from src.mit_semseg.utils import AverageMeter, accuracy, intersectionAndUnion

# import yolov3 stuff:
import src.obj_detector.detect as detect
from src.obj_detector.models import load_model
from src.obj_detector.utils.utils import load_classes

# mtcnn:
from src.mtcnn import detector as mtcnn_detector
from src.mtcnn import visualization_utils as mtcnn_utils
from PIL import Image

# import mAP eval:
from src.evaluators.map_metric.lib.BoundingBoxes import BoundingBox
from src.evaluators.map_metric.lib import BoundingBoxes
from src.evaluators.map_metric.lib.Evaluator import *
from src.evaluators.map_metric.lib.utils import BBFormat

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

class MTCNN(Model):
    def __init__(self, *network_config) -> None:
        super().__init__(*network_config)
        self.network_config = network_config
        self.complexOutput = False
    
    def run(self, input):
        _img = Image.fromarray(input)
        detections, _ = mtcnn_detector.detect_faces(self.model, _img)
        cls_matrix = np.zeros((detections.shape[0],1))
        detections = np.concatenate((detections, cls_matrix), axis=1)
        return detections

    def initialize(self):
        self.model = mtcnn_detector.load_model(*self.network_config)

    def deinitialize(self):
        return -1

    def draw(self, pred, img):
        _img = Image.fromarray(img)
        _img = mtcnn_utils.show_bboxes(_img, pred)
        img = np.array(_img)
        return {"dst": img, "listOfNames": {"all": [255,255,255], "face": [255,0,0]}}

    def draw_single_class(self, pred, img, selected_class):
        img = self.draw(pred, img)["dst"]
        return {"overlay": img}

    def report_accuracy(self, pred:list, gt:list, evalType='voc'):
        """Function takes in prediction boxes and ground truth boxes and
        returns the mean average precision (mAP) @ IOU 0.5 under VOC2007 criteria (default).

        Args:
            pred (list): A list of BoundingBox objects representing each detection from method
            gt (list): A list of BoundingBox objects representing each object in the ground truth

        Returns:
            mAP: a number representing the mAP over all classes for a single image.
        """        
        if len(pred) == 0: return 0

        allBoundingBoxes = BoundingBoxes()
        evaluator = Evaluator()

        # loop through gt:
        for _gt in gt:
            assert type(_gt) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_gt)))
            allBoundingBoxes.addBoundingBox(_gt)

        for _pred in pred:
            assert type(_pred) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_pred)))
            allBoundingBoxes.addBoundingBox(_pred)

        #for box in allBoundingBoxes:
        #    print(box.getAbsoluteBoundingBox(format=BBFormat.XYWH), box.getBBType()) 
        if evalType == 'voc':
            metrics = evaluator.GetPascalVOCMetrics(allBoundingBoxes)
        elif evalType == 'coco':
            assert False
        else: assert False, "evalType %s not supported"%(evalType) 
        return metrics[0]['AP']

    def outputFormat(self):
        return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"

class Segmentation(Model):
    """
    Segmentation Model that inhertes the Model class
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

        #color_palette = get_color_palette(pred, org_pred_split.shape[0], self.names, self.colors, detectedNames)

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
    YOLO Model that inhertes the Model class
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
        self.img_size = 416
        self.conf_thres = 0.5
        self.nms_thres = 0.5

    def run(self, input):
        pred = detect.detect_image(self.yolo, input, img_size=self.img_size, conf_thres=self.conf_thres, nms_thres=self.nms_thres)
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

    def report_accuracy(self, pred:list, gt:list, evalType='voc'):
        """Function takes in prediction boxes and ground truth boxes and
        returns the mean average precision (mAP) @ IOU 0.5 under VOC2007 criteria (default).

        Args:
            pred (list): A list of BoundingBox objects representing each detection from method
            gt (list): A list of BoundingBox objects representing each object in the ground truth

        Returns:
            mAP: a number representing the mAP over all classes for a single image.
        """        
        if len(pred) == 0: return 0

        allBoundingBoxes = BoundingBoxes()
        evaluator = Evaluator()

        # loop through gt:
        for _gt in gt:
            assert type(_gt) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_gt)))
            allBoundingBoxes.addBoundingBox(_gt)

        for _pred in pred:
            assert type(_pred) == BoundingBox, "_gt is not BoundingBox type. Instead is %s"%(str(type(_pred)))
            allBoundingBoxes.addBoundingBox(_pred)

        image = np.zeros((1400,1607,3), dtype=np.uint8)
        out_image = allBoundingBoxes.drawAllBoundingBoxes(image, '100faces')
        cv2.imwrite('test.png', out_image)

        #for box in allBoundingBoxes:
        #    print(box.getAbsoluteBoundingBox(format=BBFormat.XYWH), box.getBBType()) 
        if evalType == 'voc':
            metrics = evaluator.GetPascalVOCMetrics(allBoundingBoxes)
            print(metrics)
        elif evalType == 'coco':
            assert False
        else: assert False, "evalType %s not supported"%(evalType) 
        return metrics[0]['AP']

    def outputFormat(self):
        return "{5:.0f} {4:f} {0:.0f} {1:.0f} {2:.0f} {3:.0f}"

_registry = {
    'Face Detection (YOLOv3)': YOLOv3(
         os.path.join(currPath, 'obj_detector/cfg', 'face.names'),
         os.path.join(currPath, 'obj_detector/cfg', 'yolov3-face.cfg'),
         os.path.join(currPath,'obj_detector/weights','yolov3-face_last.weights')
    ),
    # 'Face Detection (MTCNN)': MTCNN(
    #     os.path.join(currPath, 'mtcnn/weights', 'pnet.npy'),
    #     os.path.join(currPath, 'mtcnn/weights', 'rnet.npy'),
    #     os.path.join(currPath,'mtcnn/weights','onet.npy')
    # ),
    'Semantic Segmentation': Segmentation(
        str(Path(__file__).parent.absolute()) + "/mit_semseg/config/ade20k-hrnetv2.yaml",
        scipy.io.loadmat(str(Path(__file__).parent.absolute()) + '/data/color150.mat')['colors']
    ),
    'Object Detection (YOLOv3)': YOLOv3(
        os.path.join(currPath, 'obj_detector/cfg', 'face.names'),
        os.path.join(currPath, 'obj_detector/cfg', 'yolov3.cfg'),
        os.path.join(currPath,'obj_detector/weights','yolov3.weights')
    ),
}