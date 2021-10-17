import abc

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

_registry = {
    'Semantic Segmentation': 1,
    'YOLOv3': 2
}

class segmentation()