from PyQt5.QtWidgets import QDialog
from PyQt5.uic.uiparser import QtCore
from PyQt5 import uic
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

from src.transforms import AugmentationPipeline, Augmentation
import cv2
import os
import time
import numpy as np

from src.utils.images import convertCV2QT

def createExperimentName(savePath):
    _root_name = 'exp'
    _folders = os.listdir(savePath)
    if len(_folders) == 0:
        return "%s_%i"%(_root_name, 1)
    else:
        _max_index = 1
        for folder in _folders:
            _index = int(folder.split('_')[-1])
            if _index > _max_index: _max_index = _index
        return "%s_%i"%(_root_name, _max_index+1)

class ExperimentConfig:
    def __init__(self, mainAug:AugmentationPipeline, isCompound:bool, imagePaths:list, model, shouldAug=True, labels=[]) -> None:
        self.mainAug = mainAug
        self.isCompound = isCompound
        self.imagePaths = imagePaths
        self.model = model
        self.shouldAug = shouldAug
        self.labels = labels
        self.expName = ''
        self.savePath = './src/data/tmp/runs'

class ExperimentWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    logProgress = pyqtSignal(str)

    def __init__(self, config, savePath) -> None:
        super(ExperimentWorker, self).__init__()
        self.config = config
        self.savePath = savePath

    def writeDets(self, detections, exp_path, filename):
        _file = filename.split('/')[-1]
        _txt_file = "%s.txt"%_file.split('.')[0]
        if self.config.model.complexOutput: # for multi-dimensional, complex matrices
            _format = self.config.model.outputFormat()
            detections = detections.tobytes()
            with open( os.path.join(exp_path, _txt_file), 'wb') as f:
                f.write(detections)
        else:
            _format = self.config.model.outputFormat() + '\n'
            with open( os.path.join(exp_path, _txt_file), 'w') as f:
                for det in detections:
                    f.write(_format.format(*det))

    def run(self):
        # create experiment name automatically:
        #exp_path = self.createExperimentName()
        exp_path = self.config.expName
        os.mkdir( os.path.join(self.savePath, exp_path) )
        #self.insertLog('Saving detections at: %s'%(exp_path))
        self.logProgress.emit("Saving detections at: %s"%(exp_path))

        if not self.config.shouldAug:
            for i, imgPath in enumerate(self.config.imagePaths):
                _img = cv2.imread(imgPath)
                dets = self.config.model.run(_img)
                self.writeDets(dets, os.path.join(self.savePath, exp_path), imgPath)
                #self.progressBar.setValue( ((i+1)*_progressMove)*100 )
                self.progress.emit(i)
                self.logProgress.emit('Progress: (%i/%i)'%(i,len(self.config.imagePaths)))
        else:
            if self.config.isCompound:
                # apply sequentially:
                for i, imgPath in enumerate(self.config.imagePaths):
                    _img = cv2.imread(imgPath)
                    for aug in self.config.mainAug:
                        _img = aug(_img)
                    dets = self.config.model.run(_img)
                    self.writeDets(dets, os.path.join(self.savePath, exp_path), imgPath)
                    #self.insertLog('Progress: (%i/%i)'%(i,len(self.config.imagePaths)))
                    self.logProgress.emit('Progress: (%i/%i)'%(i,len(self.config.imagePaths)))
                    #self.progressBar.setValue( ((i+1)*_progressMove)*100 )
                    self.progress.emit(i)
            else:
                for aug in self.config.mainAug:
                    # create sub folders:
                    new_sub_dir = os.path.join(exp_path, "_".join(aug.title.split(' ')) )
                    os.mkdir(new_sub_dir)
                    self.logProgress.emit('Augmentation: %s'%(aug.title))

                    for i, imgPath in enumerate(self.config.imagePaths):
                        _img = cv2.imread(imgPath)
                        # apply aug to _img here
                        dets = self.config.model.run(_img)
                        self.writeDets(dets, os.path.join(self.savePath, new_sub_dir), imgPath)
                        #self.insertLog('Progress: (%i/%i)'%(i,len(self.config.imagePaths)))
                        self.logProgress.emit('\tProgress: (%i/%i)'%(i,len(self.config.imagePaths)))
                        self.progress.emit(i)
        self.finished.emit()

class ExperimentResultWorker(QObject):
    finished = pyqtSignal()
    finishedImage = pyqtSignal(QPixmap)
    finishedGraph = pyqtSignal(np.ndarray)

    def __init__(self, imagePath, config, expName) -> None:
        super(ExperimentResultWorker, self).__init__()
        self.imagePath = imagePath # a single image
        self.config = config
        self.expName = expName

    #@QtCore.pyqtSlot()
    def run(self):
        _img = cv2.imread(self.imagePath)

        if self.config.isCompound:
            for aug in self.config.mainAug:
                _img = aug(_img)
        
        # read in detection:
        _imgRoot = self.imagePath.split('/')[-1].split('.')[0]
        txt_file = os.path.join(self.config.savePath, self.expName, "%s.txt"%(_imgRoot))
        assert os.path.exists(txt_file), txt_file

        if self.config.model.complexOutput:
            with open(txt_file, 'rb') as f:
                _bytes = f.read()
                _complex = np.frombuffer(_bytes)
            print(_complex)
            exit()
        else:
            # detectors (non-complex data)
            with open(txt_file, 'r' ) as f:
                _dets = list(map(str.strip, f.readlines()))
            dets = []
            for d in _dets:
                _d = d.split(' ')
                dets.append([ int(_d[0]), float(_d[1]), int(_d[2]), int(_d[3]), int(_d[4]), int(_d[5]) ])
            
            # apply bbox:
            for d in dets:
                _img = cv2.rectangle(_img, (d[2], d[3]), (d[4], d[5]), (0,0,255), thickness=3)

        self.finishedImage.emit(convertCV2QT(_img, 301, 211))
        self.finished.emit()

    #@QtCore.pyqtSlot()
    def runGraph(self):
        # Priya code here
        return 0

class ExperimentDialog(QDialog):
    def __init__(self, config:ExperimentConfig) -> None:
        super(ExperimentDialog, self).__init__()
        uic.loadUi('./src/qt_designer_file/experiment.ui', self)
        self.__setPreviews__(False)
        self.progressBar.setValue(0)
        self.textProgress.setEnabled(False)
        self.config = config
        self._progressMove = 1/len(self.config.imagePaths)
        self.config.expName = createExperimentName(self.config.savePath)

        # image gui controls:
        self.currentIdx = 0
        self.currentGraphIdx = 0
        self.totalGraphs = 1

        # buttons:
        self.previewBack.clicked.connect(lambda: self.changeOnImageButton(-1)) # substract one from currentIdx
        self.previewForward.clicked.connect( lambda: self.changeOnImageButton(1) ) # increase index by one

        # multithreading stuff for updates after experiment:
        self.afterExpThread = QThread()

        # check if experiment folder exists:
        if not os.path.exists(self.config.savePath):
            os.mkdir(self.config.savePath)
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
        
        # Opposite:
        self.progressBar.setVisible(not state)
        self.textProgress.setVisible(not state)

    def insertLog(self, text):
        self.textProgress.insertPlainText('%s\n'%(text))
        return 0

    def startExperiment(self):
        self.textProgress.clear()

        self.thread = QThread()
        self.worker = ExperimentWorker(self.config, self.config.savePath)
        if not self.config.isCompound:
            self.totalGraphs = len(self.config.mainAug)

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.displayResults)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect( lambda i: self.progressBar.setValue( ((i+1)*self._progressMove)*100 ) )
        self.worker.logProgress.connect(self.insertLog)

        # Step 6: Start the thread
        self.thread.start()

    def displayResults(self):
        self.thread.quit()

        # update metadata on the labels:
        self.label_3.setText(str(len(self.config.imagePaths)))
        self.label_6.setText(str(self.totalGraphs))
        self.label.setText(str(self.currentIdx+1))
        self.label_4.setText(str(self.currentGraphIdx+1))

        self.__setPreviews__(True)
        self.refreshImageResults(0)
        self.refreshGraphResults(0)

    def refreshImageResults(self,i):
        self.worker = ExperimentResultWorker(self.config.imagePaths[i], self.config, self.config.expName)
        self.worker.moveToThread(self.afterExpThread)
        self.afterExpThread.started.connect(self.worker.run)
        self.worker.finished.connect(self.afterExpThread.quit)
        self.worker.finished.connect(self.afterExpThread.wait)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finishedImage.connect(self.updateImage)
        #self.afterExpThread.finished.connect(self.afterExpThread.deleteLater)
        #worker.progress.connect()
        self.afterExpThread.start()
    
    def updateImage(self, img):
        self.previewImage.setPixmap(img)

    def updateGraph(self,):
        return 

    def refreshGraphResults(self,i):
        return i
        worker = ExperimentResultWorker(self.config.imagePaths[i], self.config, self.config.expName)
    
    def changeOnImageButton(self, i):
        if self.currentIdx+i < len(self.config.imagePaths) and self.currentIdx+i >= 0:
            self.currentIdx += i
            self.label.setText(str(self.currentIdx+1))
            self.refreshImageResults(self.currentIdx)

    def changeOnGraphButton(self, i):
        if self.currentGraphIdx+i < len(self.config.imagePaths) and self.currentGraphIdx+i >= 0:
            self.currentGraphIdx += i
            self.label.setText(str(self.currentGraphIdx+1))
            self.refreshGraphResults(self.currentGraphIdx)
            return 0