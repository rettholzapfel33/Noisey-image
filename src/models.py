import abc
import os
from pathlib import Path
from PyQt5.QtCore import QObject
import os, csv, torch, scipy.io

from src.predict_img import process_img, predict_img, load_model_from_cfg, visualize_result, transparent_overlays, get_color_palette

# import yolov3 stuff:
import src.obj_detector.detect as detect
from src.obj_detector.models import load_model
from src.obj_detector.utils.utils import load_classes

currPath = str(Path(__file__).parent.absolute()) + '/'

class Model(abc.ABC):
    def __init__(self, *network_config) -> None:
        self.__network_config__ = network_config
        self.initialize(*network_config)
    
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
    def draw(pred):
        raise NotImplementedError

    def __call__(self):
        pred = self.run()
        return pred

class Segmentation(Model):
    def __init__(self, *network_config) -> None:
        self.cfg = str(Path(__file__).parent.absolute()) + "/config/ade20k-hrnetv2.yaml"
        # colors
        self.colors = scipy.io.loadmat(str(Path(__file__).parent.absolute()) + '/data/color150.mat')['colors']
        self.names = {}
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

    

class YOLOv3(Model):
    def __init__(self, *network_config) -> None:
        # network_config: CLASSES, CFG, WEIGHTS
        #self.CLASSES, self.CFG, self.WEIGHTS = network_config
        self.CLASSES = os.path.join(currPath, 'obj_detector/cfg', 'coco.names')
        self.CFG = os.path.join(currPath, 'obj_detector/cfg', 'yolov3.cfg')
        self.WEIGHTS = os.path.join(currPath,'obj_detector/weights','yolov3.weights')
        print(self.CLASSES, self.CFG, self.WEIGHTS)

    def run(self, input):
        pred = detect.detect_image(self.yolo, input)
        
        return pred

    def initialize(self, *kwargs):
        self.yolo = load_model(self.CFG, self.WEIGHTS)
        self.classes = load_classes(self.CLASSES)
        return 0
    
    def deinitialize(self):
        return -1
    
    def draw(self, pred, img):
        np_img = detect._draw_and_return_output_image(img, pred, 416, self.classes)
        return {"dst": np_img}

_registry = {
    'Semantic Segmentation': 1,
    'YOLOv3': YOLOv3(
        os.path.join(currPath, '/obj_detector/cfg', 'coco.names'),
        os.path.join(currPath, '/obj_detector/cfg', 'yolov3.cfg'),
        os.path.join(currPath,'/obj_detector/weights','yolov3.weights')
    )
}