import abc
import os
from pathlib import Path
from PyQt5.QtCore import QObject

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
        pass

    def run(self, input):
        return 0

    def initialize(self, *kwargs):
        return 0

    def deinitialize(self):
        return -1

    def draw(pred):
        return 0

class YOLOv3(Model):
    def __init__(self, *network_config) -> None:
        # network_config: CLASSES, CFG, WEIGHTS
        self.CLASSES, self.CFG, self.WEIGHTS = network_config
        print(self.CLASSES, self.CFG, self.WEIGHTS)

    def run(self, input):
        return 0

    def initialize(self, *kwargs):
        return 0
    
    def deinitialize(self):
        return -1
    
    def draw(pred):
        return 0

_registry = {
    'Semantic Segmentation': 1,
    'YOLOv3': YOLOv3(
        os.path.join(currPath, '/obj_detector/cfg', 'coco.names'),
        os.path.join(currPath, '/obj_detector/cfg', 'yolov3.cfg'),
        os.path.join(currPath,'/obj_detector/weights','yolov3.weights')
    )
}