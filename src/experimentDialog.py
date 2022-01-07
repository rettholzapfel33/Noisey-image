from PyQt5.QtWidgets import QDialog
from PyQt5.uic.uiparser import QtCore
from PyQt5 import uic

from src.transforms import AugmentationPipeline, Augmentation

class ExperimentDialog(QDialog):
    def __init__(self) -> None:
        super(ExperimentDialog, self).__init__()
        uic.loadUi('./src/qt_designer_file/experiment.ui', self)

    def startExperiment(self, mainAug:AugmentationPipeline, isComposed: bool, imagePaths: list, ):
        if isComposed:
            # apply sequentially:
            pass
        return 0