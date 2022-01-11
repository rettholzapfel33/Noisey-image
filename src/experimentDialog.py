from PyQt5.QtWidgets import QDialog
from PyQt5.uic.uiparser import QtCore
from PyQt5 import uic

from src.transforms import AugmentationPipeline, Augmentation
import cv2
import os
import time

class ExperimentConfig:
    def __init__(self, mainAug:AugmentationPipeline, isCompound:bool, imagePaths:list, infer, shouldAug=True) -> None:
        self.mainAug = mainAug
        self.isCompound = isCompound
        self.imagePaths = imagePaths
        self.infer = infer
        self.shouldAug = shouldAug

class ExperimentDialog(QDialog):
    def __init__(self, config:ExperimentConfig) -> None:
        super(ExperimentDialog, self).__init__()
        uic.loadUi('./src/qt_designer_file/experiment.ui', self)
        self.__setPreviews__(False)
        self.progressBar.setValue(0)
        self.textProgress.setEnabled(False)
        self.config = config
        self.savePath = './src/data/tmp/runs'
        # check if experiment folder exists:
        if not os.path.exists(self.savePath):
            os.mkdir(self.savePath)
        self.show()

    def __setPreviews__(self, state:bool):
        self.label_6.setVisible(state)
        self.label_5.setVisible(state)
        self.label_4.setVisible(state)
        self.label_3.setVisible(state)
        self.label_2.setVisible(state)
        self.label.setVisible(state)
        self.previewBack.setVisible(state)
        self.previewForward.setVisible(state)
        self.backGraph.setVisible(state)
        self.forwardGraph.setVisible(state)
        self.graphImage.setVisible(state)
        self.previewImage.setVisible(state)

    def insertLog(self, text):
        self.textProgress.insertPlainText('%s\n'%(text))
        return 0

    def createExperimentName(self):
        _root_name = 'exp'
        _folders = os.listdir(self.savePath)
        if len(_folders) == 0:
            return "%s_%i"%(_root_name, 1)
        else:
            _max_index = 1
            for folder in _folders:
                _index = int(folder.split('_')[-1])
                if _index > _max_index: _max_index = _index
            return "%s_%i"%(_root_name, _max_index+1)

    def loadConfig(self, config:ExperimentConfig):
        self.config = config
        print("Configuration loaded!")

    def writeDets(self, detections, exp_path, filename):
        _file = filename.split('/')[-1]
        _txt_file = "%s.txt"%_file.split('.')[0]
        with open( os.path.join(exp_path, _txt_file), 'w') as f:
            for det in detections:
                f.write('%i %f %i %i %i %i\n'%(det[0], det[1], det[2], det[3], det[4], det[5]))

    def startExperiment(self):
        self.textProgress.clear()
        # create experiment name automatically:
        exp_path = self.createExperimentName()
        os.mkdir( os.path.join(self.savePath, exp_path) )
        self.insertLog('Saving detections at: %s'%(exp_path))
        _progressMove = 1/len(self.config.imagePaths)

        if self.config.shouldAug:
            for i, imgPath in enumerate(self.config.imagePaths):
                _img = cv2.imread(imgPath)
                dets = self.config.infer(_img)
        else:
            if self.config.isCompound:
                # apply sequentially:
                for i, imgPath in enumerate(self.config.imagePaths):
                    _img = cv2.imread(imgPath)
                    for aug in self.config.mainAug:
                        print(aug)
                    dets = self.config.infer(_img)
                    self.writeDets(dets, exp_path, imgPath)
                    self.insertLog('Progress: (%i/%i)'%(i,len(self.config.imagePaths)))
                    self.progressBar.setValue( ((i+1)*_progressMove)*100 )
            else:
                for aug in self.config.mainAug:
                    # create sub folders:
                    new_sub_dir = os.path.join(exp_path, "_".join(aug.title.split(' ')) )
                    os.mkdir(new_sub_dir)

                    for i, imgPath in enumerate(self.config.imagePaths):
                        _img = cv2.imread(imgPath)
                        # apply aug to _img here
                        dets = self.config.infer(_img)
                        self.writeDets(dets, new_sub_dir, imgPath)
                        self.insertLog('Progress: (%i/%i)'%(i,len(self.config.imagePaths)))
                        self.progressBar.setValue( ((i+1)*_progressMove)*100 )