from PyQt5.QtWidgets import QDialog
from PyQt5.uic.uiparser import QtCore
from PyQt5 import uic
from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
#from charset_normalizer import detect
from src.mplwidget import MplWidget
import matplotlib.pyplot as plt

from src.transforms import AugmentationPipeline, Augmentation
import cv2
import os
import time
import numpy as np
import yaml

from src.utils.images import convertCV2QT
import matplotlib.pyplot as plt

# calculating stats:
from src.evaluators.map_metric import driver

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
    def __init__(self, mainAug:AugmentationPipeline, isCompound:bool, imagePaths:list, model, modelName, shouldAug=True, labels=[]) -> None:
        self.mainAug = mainAug
        self.isCompound = isCompound
        self.imagePaths = imagePaths
        self.model = model
        self.modelName = modelName
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

    def writeMeta(self, outPath):
        with open(os.path.join(outPath, 'meta.yaml'), 'w') as f:
            _out = {}
            for aug in self.config.mainAug:
                _out[aug.title] = aug.args
            yaml.dump(_out, f)

    def writeGraph(self, inData, outPath):
       #np.savetxt(os.path.join(outPath, 'graphing.csv'),inData,)
       np.save(os.path.join(outPath, 'graphing.npy'), inData)

    def calculateStat(self, dets, assembler, i):
        if self.config.modelName == 'Object Detection (YOLOv3)':
            if len(self.config.labels) == 0:
                if assembler is None: assembler = 0
                assembler += len(dets)
            else:
                # do mAP calculation here
                pass
        elif self.config.modelName == 'Semantic Segmentation':
            if len(self.config.labels) == 0:
                if assembler is None: assembler = []
                ratios = self.config.model.calculateRatios(dets)
                total_pixels = np.sum(ratios)
                ratios = (ratios / total_pixels)*100
                assembler.append(ratios)
            else:
                pass # do whatever segmentation needs for eval LOL IDK
        elif self.config.modelName == 'EfficientNetV2':
            if len(self.config.labels) == 0:
                if assembler is None: assembler = 0
                assembler += len(dets)
            else:
                pass
        else: raise Exception('model name is not recognized in _registry'%(self.config.modelName))
        
        print(assembler)
        return assembler

    def run(self):
        # create experiment name automatically:
        exp_path = self.config.expName
        os.mkdir( os.path.join(self.savePath, exp_path) )
        self.logProgress.emit("Saving detections at: %s"%(exp_path))
        
        # write meta out for later loading and reference
        self.writeMeta(os.path.join(self.savePath, exp_path))

        if len(self.config.mainAug) == 0:
            for i, imgPath in enumerate(self.config.imagePaths):
                _img = cv2.imread(imgPath)
                dets = self.config.model.run(_img)
                self.writeDets(dets, os.path.join(self.savePath, exp_path), imgPath)
                self.logProgress.emit('\tProgress: (%i/%i)'%(i,len(self.config.imagePaths)))
                self.progress.emit(i)
        else:
            if self.config.isCompound:
                # apply sequentially (all args must be of the same length):
                maxArgLen = len(self.config.mainAug.__pipeline__[0].args)
                # create variables for simple counting rather than mAP calculation:
                counter = []
                for j in range(maxArgLen):
                    _count = None
                    for i, imgPath in enumerate(self.config.imagePaths):
                        self.logProgress.emit("Running column %i of Augmentations"%(j))
                        j_subFolder = '_'.join(["_".join(aug.title.split(" ")) for aug in self.config.mainAug])
                        j_subFolder += '_'+str(j)

                        try: os.mkdir( os.path.join(self.savePath, exp_path, j_subFolder) )
                        except FileExistsError: print("Folder path already exists...")

                        _img = cv2.imread(imgPath)
                        
                        for aug in self.config.mainAug:
                            _args = aug.args
                            _img = aug(_img, _args[j])
        
                        dets = self.config.model.run(_img)
                        _count = self.calculateStat(dets, _count, i)

                        self.writeDets(dets, os.path.join(self.savePath, exp_path, j_subFolder), imgPath)
                        self.logProgress.emit('Progress: (%i/%i)'%(i,len(self.config.imagePaths)))
                        self.progress.emit(i)

                    print(_count)
                    if type(_count) == int: _count /= len(self.config.imagePaths)
                    counter.append(_count)
                counter = [counter]
            else:
                # create variables for simple counting rather than mAP calculation:
                counter = []
                for aug in self.config.mainAug:
                    count_temp = []
                    self.logProgress.emit('Augmentation: %s'%(aug.title))
                    
                    for j in range(len(aug.args)):
                        _count = None
                        # create subdirectory here for each augmentation
                        new_sub_dir = os.path.join(self.config.savePath, exp_path, "%s_%i"%(
                            "_".join(aug.title.split(' ')), j)
                        )
                        try: os.mkdir(new_sub_dir)
                        except FileExistsError: print("Folder already exists...")

                        for i, imgPath in enumerate(self.config.imagePaths):
                            _img = cv2.imread(imgPath)
                            _img = aug(_img, request_param=aug.args[j])
                            dets = self.config.model.run(_img)
                            self.writeDets(dets, new_sub_dir, imgPath)
                            #self.insertLog('Progress: (%i/%i)'%(i,len(self.config.imagePaths)))
                            self.logProgress.emit('\tProgress: (%i/%i)'%(i,len(self.config.imagePaths)))
                            self.progress.emit(i)
                            _count = self.calculateStat(dets, _count, i)

                        if type(_count) == int: _count /= len(self.config.imagePaths)
                        count_temp.append(_count)
                    counter.append(count_temp)

        print(counter)
        #exit() 
        self.writeGraph(counter, os.path.join(self.savePath, exp_path))
        # clean up model
        self.config.model.deinitialize()
        self.finished.emit()

class ExperimentResultWorker(QObject):
    finished = pyqtSignal()
    finishedImage = pyqtSignal(QPixmap)
    finishedGraph = pyqtSignal(list)

    def __init__(self, imagePath, config, expName, augPosition=None, argPosition=None) -> None:
        super(ExperimentResultWorker, self).__init__()
        self.imagePath = imagePath # a single image
        self.config = config
        self.expName = expName
        self.augPosition = augPosition # only used when augmentations are not compounded, need 2 know position of wanted aug
        self.argPosition = argPosition # only used when aug's compounded
        self.parentPath = os.path.join(self.config.savePath, self.expName)
        self.folders = next(os.walk(self.parentPath))[1]
    
    def run(self):
        if self.config.isCompound:
            _folder_path = os.path.join(self.parentPath, self.folders[self.argPosition])
        elif not self.config.isCompound:
            _title = "_".join(self.config.mainAug.__pipeline__[self.augPosition].title.split(" "))
            _folder_path = os.path.join(self.parentPath, "%s_%i"%(_title, self.argPosition))
        elif len(self.config.mainAug.__pipeline__) == 0:
            _folder_path = self.parentPath

        _img = cv2.imread(self.imagePath)

        if len(self.config.mainAug) > 0:
            if self.config.isCompound:
                for aug in self.config.mainAug:
                    _img = aug(_img, aug.args[self.argPosition])
            else:
                aug = self.config.mainAug.__pipeline__[self.augPosition]
                _img = aug(_img, aug.args[self.argPosition])
        
        # read in detection:
        _imgRoot = self.imagePath.split('/')[-1].split('.')[0]
        txt_file = os.path.join(_folder_path, "%s.txt"%(_imgRoot))
        assert os.path.exists(txt_file), txt_file

        if self.config.model.complexOutput:
            with open(txt_file, 'rb') as f:
                _bytes = f.read()
                _complex = np.frombuffer(_bytes, dtype=np.int64)
            _complex = _complex.reshape(_img.shape[:2])
            _img_dict = self.config.model.draw(_complex, _img)
            _img = _img_dict['dst']
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

        self.finishedImage.emit(convertCV2QT(_img, 391, 231))
        self.finished.emit()

    def runGraph(self):
        fig, ax = plt.subplots(1,1)
        with open( os.path.join(self.parentPath, 'meta.yaml') ) as stream:
            try:
                graphContent = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        _graphs = np.load(os.path.join(self.parentPath, 'graphing.npy'), allow_pickle=True)
        print(_graphs)
        #exit()

        if self.config.modelName == 'Semantic Segmentation':
            _graphs = _graphs.squeeze(2)

        if self.config.isCompound:
            # segmentation specific stuff:
            if self.config.modelName == 'Semantic Segmentation':
                _graphs = _graphs.squeeze(0)
                _graphs = np.transpose(_graphs, (1,0))

            _title = ", ".join(list(graphContent.keys()))
            print(_title)
            _items = np.array(list(graphContent.values()))
            argLen = len(_items[0])
            #_g = _graphs[self.argPosition]
            _x = [i for i in range(argLen)]
            #_x = [",".join(_items[:, i]) for i in range(argLen)] # might be slow
            ax.set_title(_title)

            for _g in _graphs:
                print(_x, _g, argLen)
                ax.plot(_x, _g, 'o-')
        else:
            _g = _graphs[self.augPosition]
            _keys = list(graphContent.keys())
            _items = np.array(list(graphContent.values()))
            _title = _keys[self.augPosition]
            #_x = [i for i in range(len(_items[self.argPosition]))]
            _x = _items[self.argPosition]
            ax.set_title(_title)
            ax.plot(_x, _g, '-o')
        self.finishedGraph.emit([fig, ax])
        self.finished.emit()

class ExperimentDialog(QDialog):
    def __init__(self, config:ExperimentConfig) -> None:
        super(ExperimentDialog, self).__init__()
        uic.loadUi('./src/qt_designer_file/experiment.ui', self)
       
        # create graph widget in here:
        self.graphWidget = MplWidget()
        self.graphWidget.resize(481, 301)
        self.graphWidget.move(390,50)
        self.graphGrid.addWidget(self.graphWidget)

        self.progressBar.setValue(0)
        self.textProgress.setEnabled(False)
        self.config = config
        self._progressMove = 1/len(self.config.imagePaths)
        self.config.expName = createExperimentName(self.config.savePath)

        # image gui controls:
        self.currentIdx = 0
        self.currentArgIdx = 0
        self.currentGraphIdx = 0
        # total amount of garphs, arguments, augmentations
        self.totalGraphs = 1
        self.totalArgIdx = 0

        # fill in combobox:
        self.totalArgIdx = len(self.config.mainAug.__pipeline__[0].args)
        if self.config.isCompound: 
            self.augComboBox.setVisible(False)
        else:
            for i, aug in enumerate(self.config.mainAug):
                self.augComboBox.addItem(aug.title)
            self.augComboBox.currentIndexChanged.connect(lambda: self.refreshImageResults(self.currentIdx))
            self.augComboBox.currentIndexChanged.connect(lambda: self.refreshGraphResults(self.currentIdx))

        # buttons:
        self.previewBack.clicked.connect(lambda: self.changeOnImageButton(-1) ) # substract one from currentIdx
        self.previewForward.clicked.connect( lambda: self.changeOnImageButton(1) ) # increase index by one
        self.previewBack_3.clicked.connect(lambda: self.changeOnImageAugButton(-1) )
        self.previewForward_3.clicked.connect(lambda: self.changeOnImageAugButton(1) )
        #self.forwardGraph.clicked.connect(lambda: self.changeOnGraphButton(1))
        #self.backGraph.clicked.connect(lambda: self.changeOnGraphButton(-1))

        # multithreading stuff for updates after experiment:
        self.afterExpThread = QThread()
        self.afterGraphExpThread = QThread()

        # check if experiment folder exists:
        if not os.path.exists(self.config.savePath):
            os.mkdir(self.config.savePath)

        self.__setPreviews__(False)
        self.show()

    def __setPreviews__(self, state:bool):
        if self.config.isCompound: self.augComboBox.setVisible(False)
        else: self.augComboBox.setVisible(state)
        self.degrade_label.setVisible(state)
        self.image_label.setVisible(state)
        self.label_7.setVisible(state)
        self.label_11.setVisible(state)
        self.label_12.setVisible(state)
        self.label_13.setVisible(state)
        self.previewBack_3.setVisible(state)
        self.previewForward_3.setVisible(state)
        #self.label_6.setVisible(state)
        #self.label_5.setVisible(state)
        #self.label_4.setVisible(state)
        self.label_3.setVisible(state)
        self.label_2.setVisible(state)
        self.label.setVisible(state)
        self.previewBack.setVisible(state)
        self.previewForward.setVisible(state)
        #self.backGraph.setVisible(state)
        #self.forwardGraph.setVisible(state)
        #self.graphImage.setVisible(state)
        self.previewImage.setVisible(state)
        self.graphWidget.setVisible(state)
        
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
        #self.label_6.setText(str(self.totalGraphs))
        self.label.setText(str(self.currentIdx+1))
        #self.label_4.setText(str(self.currentGraphIdx+1))
        self.label_13.setText(str(self.totalArgIdx))
        self.label_11.setText(str(self.currentArgIdx+1))

        self.__setPreviews__(True)
        self.refreshImageResults(0)
        self.refreshGraphResults(0)

    def refreshImageResults(self,i):
        augPosition = self.augComboBox.currentIndex()
        if self.config.isCompound:
            self.worker = ExperimentResultWorker(self.config.imagePaths[i], self.config, self.config.expName, argPosition=self.currentArgIdx)
        else:
            self.totalArgIdx = len(self.config.mainAug.__pipeline__[augPosition].args)
            if self.currentArgIdx >= self.totalArgIdx:
                self.currentArgIdx = self.totalArgIdx-1
                #i = self.currentIdx
                self.label_11.setText(str(self.currentArgIdx+1))
            self.worker = ExperimentResultWorker(self.config.imagePaths[i], self.config, self.config.expName, argPosition=self.currentArgIdx, augPosition=augPosition)
            self.label_13.setText(str(self.totalArgIdx))

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

    def updateGraph(self, ax_list):
        fig, ax = ax_list
        for i in range(len(ax.lines)):
            line = ax.lines[i]
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            # self.graphWidget.canvas.axes.clear()
            self.graphWidget.canvas.axes.plot(x_data, y_data, 'o-')
        self.graphWidget.canvas.axes.set_title(ax.get_title())
        self.graphWidget.canvas.axes.set_xlabel(str(ax.xaxis.get_label()))
        self.graphWidget.canvas.axes.set_ylabel(str(ax.yaxis.get_label()))
        self.graphWidget.canvas.draw()

    def refreshGraphResults(self,i):
        augPosition = self.augComboBox.currentIndex()
        #worker = ExperimentResultWorker(self.config.imagePaths[i], self.config, self.config.expName)
        # graphing:
        self.workerGraph = ExperimentResultWorker(None, self.config, self.config.expName, argPosition=self.currentArgIdx, augPosition=augPosition)
        self.afterGraphExpThread.started.connect(self.workerGraph.runGraph)
        self.workerGraph.finished.connect(self.afterGraphExpThread.quit)
        self.workerGraph.finished.connect(self.afterGraphExpThread.wait)
        self.workerGraph.finished.connect(self.workerGraph.deleteLater)
        self.workerGraph.finishedGraph.connect(self.updateGraph)
        #self.worker.finishedImage.connect(self.updateImage)
        #self.afterExpThread.finished.connect(self.afterExpThread.deleteLater)
        #worker.progress.connect()
        self.afterGraphExpThread.start()

    def changeOnImageButton(self, i):
        if self.currentIdx+i < len(self.config.imagePaths) and self.currentIdx+i >= 0:
            self.currentIdx += i
            self.label.setText(str(self.currentIdx+1))
            self.refreshImageResults(self.currentIdx)

    def changeOnImageAugButton(self, i):
        if self.currentArgIdx+i < self.totalArgIdx and self.currentArgIdx+i >= 0:
            self.currentArgIdx += i
            self.label_11.setText(str(self.currentArgIdx+1))
            self.refreshImageResults(self.currentIdx)

    def changeOnGraphButton(self, i):
        if self.currentGraphIdx+i < self.totalGraphs and self.currentGraphIdx+i >= 0:
            self.currentGraphIdx += i
            self.label_4.setText(str(self.currentGraphIdx+1))
            self.refreshGraphResults(self.currentGraphIdx)