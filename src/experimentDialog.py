from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5 import uic
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QSize, Qt
from PyQt5.QtGui import QPixmap, QIcon, QImage
from src.mplwidget import MplWidget
import matplotlib.pyplot as plt
from multiprocessing import Value

from makeBetterGraph import makemAPGraph
from src.transforms import AugmentationPipeline, Augmentation
import cv2
import os
import re
import time
import json 
import numpy as np
import yaml

from src.utils.images import convertCV2QT
import matplotlib.pyplot as plt

# eval imports:
from src.evaluators.map_metric.lib.BoundingBoxes import BoundingBox
from src.evaluators.map_metric.lib import BoundingBoxes
from src.evaluators.map_metric.lib.Evaluator import *
from src.evaluators.map_metric.lib.utils import BBFormat

# coco api import
from pycocotools.cocoeval import COCOeval
from src.utils.coco_utils import coco80_to_coco91_class
coco_remap = coco80_to_coco91_class()

def createExperimentName(savePath):
    _root_name = 'exp'
    if not os.path.exists(savePath): os.mkdir(savePath)
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
    def __init__(self, mainAug:AugmentationPipeline, isCompound:bool, imagePaths:list, model, modelName, shouldAug=True, labels=[], labelType="voc") -> None:
        self.mainAug = mainAug
        self.isCompound = isCompound
        self.imagePaths = imagePaths
        self.model = model
        self.modelName = modelName
        self.shouldAug = shouldAug
        self.labels = labels
        self.labelType = labelType
        self.expName = ''
        self.savePath = './src/data/tmp/runs'

class ExperimentWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    logProgress = pyqtSignal(str)

    def __init__(self, config, savePath, threadDone) -> None:
        super(ExperimentWorker, self).__init__()
        self.config = config
        self.savePath = savePath 
        self.cocoJSON = ''
        self.threadDone = threadDone

    def writeDets(self, detections, exp_path, filename, outObject=None):
        # x1, y1, x2, y2, conf, cls
        # If running the COCO dataset
        if self.config.labelType == 'coco':
            assert type(outObject) == list, "outObject for labelType coco needs to be a list!"
            # Properly format the output 
            _format = self.config.model.outputFormat()
            
            # Loop through the detections
            for det in detections:
                # Split up the output into multiple strings
                split = re.split(' ', _format.format(*det))
                if not self.config.model.isCOCO91:
                    det[5] = coco_remap[int(det[5])] # convert coco91 to 80

                # Create dictionary 
                result = {
                    'image_id': filename, # filename -> the given coco imgid
                    'category_id': int(det[5]),
                    'bbox': [float(split[2]), float(split[3]), float(split[4]), float(split[5])],
                    'score': float("{:.3f}".format(float(split[1])))
                }
                outObject.append(result)
            _file = self.config.labels['coco'].loadImgs(filename)[0]['file_name']
            #_file = os.path.join(self.config.labels['root'], _file)
        else:
            _file = filename.split('/')[-1]
        _txt_file = "%s.txt"%_file.split('.')[0]
        if self.config.model.complexOutput: # for multi-dimensional, complex matrices
            _format = self.config.model.outputFormat()
            detections = detections.tobytes()
            with open( os.path.join(exp_path, _txt_file), 'wb') as f:
                f.write(detections)

        _format = self.config.model.outputFormat() + '\n'
        with open( os.path.join(exp_path, _txt_file), 'w') as f:
            for det in detections:
                #print(det)
                if self.config.labelType == 'coco':
                    if len(det) > 0:
                        det[[2,3]] += det[[0,1]]
                
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
       title = ""
       for aug in self.config.mainAug:
           title = aug.title
           break
       makemAPGraph(outPath, title, self.config.modelName)

    def calculateStat(self, dets, assembler, i, filename):
        if self.config.modelName == 'Object Detection (YOLOv3)':
            if len(self.config.labels) == 0:
                if assembler is None: assembler = 0
                assembler += len(dets)
            else:
                raise NotImplementedError()
                # do mAP calculation here
                
        elif self.config.modelName == 'Face Detection (YOLOv3)' or 'Object Detection (YOLOv3-Ultra)':
            if len(self.config.labels) == 0:
                if assembler is None: assembler = 0
                assembler += len(dets)
            else:
                if assembler is None: assembler = 0
                # convert to boundingbox classes
                #[x1,y1,x2,y2,conf,class] <--- box
                if filename in self.config.labels:
                    preds = [BoundingBox(filename, det[5], det[0], det[1], det[2]-det[0], det[3]-det[1], classConfidence=det[4], bbType=BBType.Detected) for det in dets]
                    gt = self.config.labels[filename]
                    #preds = [ BoundingBox(filename, 0, det.getAbsoluteBoundingBox(format=BBFormat.XYWH)[0], det.getAbsoluteBoundingBox(format=BBFormat.XYWH)[1], det.getAbsoluteBoundingBox(format=BBFormat.XYWH)[2], det.getAbsoluteBoundingBox(format=BBFormat.XYWH)[3], format=BBFormat.XYWH, bbType=BBType.Detected, classConfidence=0.99)  for det in self.config.labels[filename]]
                    mAP50 = self.config.model.report_accuracy(preds, gt, self.config.labelType)
                    # do mAP calculation here
                    assembler += mAP50
                else: print("WARNING: %s does not have key in labels! Ignoring for now.")
                
        elif self.config.modelName == 'Semantic Segmentation':
            if len(self.config.labels) == 0:
                if assembler is None: assembler = []
                ratios = self.config.model.calculateRatios(dets)
                total_pixels = np.sum(ratios)
                ratios = (ratios / total_pixels)*100
                if len(assembler) == 0:
                    assembler.append(ratios)
                else:
                    assembler[0] = (assembler[0]+ratios)/i
            else:
                raise NotImplementedError()
                pass # do whatever segmentation needs for eval LOL IDK
        elif self.config.modelName == 'EfficientDetV2':
            if len(self.config.labels) == 0:
                if assembler is None: assembler = 0
                assembler += len(dets)
            else:
                pass
        elif self.config.modelName == 'Object Detection (DETR)':
            if len(self.config.labels) == 0:
                if assembler is None: assembler = 0
                assembler += len(dets)
            else:
                pass
        elif self.config.modelName == 'Object Detection (YOLOv4)':
            if len(self.config.labels) == 0:
                if assembler is None: assembler = 0
                assembler += len(dets)
            else:
                pass
        else: raise Exception('model name %s is not recognized in _registry'%(self.config.modelName))
        return assembler

    def run(self):
        # create experiment name automatically:
        exp_path = self.config.expName
        os.mkdir( os.path.join(self.savePath, exp_path) )
        self.logProgress.emit("Saving detections at: %s"%(exp_path))
        
        # write meta out for later loading and reference
        self.writeMeta(os.path.join(self.savePath, exp_path))

        # map compute purposes:
        if type(self.config.labels) == list:
            if len(self.config.labels) == 0:
                useLowerThres = False
            else: useLowerThres = True
        else: useLowerThres = True

        if useLowerThres:
            old_thres = self.config.model.conf_thres
            self.config.model.conf_thres = 0.001

        if len(self.config.mainAug) == 0:
            for i, imgPath in enumerate(self.config.imagePaths):
                if self.config.labelType == 'coco':
                    imgID = imgPath
                    imgPath = self.config.labels['coco'].loadImgs(imgPath)[0]
                    imgPath = imgPath['file_name']
                    imgPath = os.path.join(self.config.labels['root'], imgPath)
                    _img = cv2.imread(imgPath)
                    dets = self.config.model.run(_img)
                    self.writeDets(dets, os.path.join(self.savePath, exp_path), imgID, outObject=_json)
                else:
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
                        if self.config.labelType == 'coco':
                            _json = []
                            _filter_id = []

                        _count = None
                        # create subdirectory here for each augmentation
                        new_sub_dir = os.path.join(self.config.savePath, exp_path, "%s_%i"%(
                            "_".join(aug.title.split(' ')), j)
                        )
                        try: os.mkdir(new_sub_dir)
                        except FileExistsError: print("Folder already exists...")

                        for i, imgPath in enumerate(self.config.imagePaths):
                            if self.config.labelType == 'coco':
                                imgID = imgPath
                                imgPath = self.config.labels['coco'].loadImgs(imgPath)[0]
                                imgPath = imgPath['file_name']
                                imgPath = os.path.join(self.config.labels['root'], imgPath)
                                _img = cv2.imread(imgPath)
                                _img = aug(_img, request_param=aug.args[j])
                                dets = self.config.model.run(_img)

                                if i > 5000: # run the first 50 images
                                    break
                                _filter_id.append(imgID)
                                
                                # convert from xyxy to xywh:
                                if len(dets) > 0:
                                    dets[:,[2,3]] = dets[:, [2,3]] - dets[:, [0,1]]  
                                    self.writeDets(dets, new_sub_dir, imgID, outObject=_json)
                            else:
                                _img = cv2.imread(imgPath)
                                _img = aug(_img, request_param=aug.args[j])
                                dets = self.config.model.run(_img)
                                self.writeDets(dets, new_sub_dir, imgPath)
                                
                            self.logProgress.emit('\tProgress: (%i/%i)'%(i,len(self.config.imagePaths)))
                            self.progress.emit(i)
                            if self.config.labelType != 'coco':
                                _count = self.calculateStat(dets, _count, i, os.path.splitext(imgPath.split('/')[-1])[0] ) # find the filename only

                        # finish COCO output and calculate mAP:
                        if self.config.labelType == 'coco':
                            self.logProgress.emit("\tCreating json output...")
                            _jsonObj = json.dumps(_json)
                            with open( os.path.join(new_sub_dir, "cocoRes.json") , "w") as outfile:
                                outfile.write(_jsonObj)
                            cocoDets = self.config.labels['coco'].loadRes( os.path.join(new_sub_dir, "cocoRes.json") )
                            cocoEval = COCOeval(self.config.labels['coco'],cocoDets,'bbox')
                            self.logProgress.emit("\tEvaluating with COCO metric...")
                            cocoEval.params.imgIds  = _filter_id
                            cocoEval.evaluate()
                            cocoEval.accumulate()
                            cocoEval.summarize()
                            map_list = cocoEval.stats
                            with open(os.path.join(new_sub_dir, "map.txt"), 'w') as f:
                                for _map in map_list:
                                    f.write("%f\n"%(_map))
                            count_temp.append(map_list[1])
                        else:
                            if type(_count) == int: _count /= len(self.config.imagePaths)
                            count_temp.append(_count)
                    counter.append(count_temp)

        #if self.config.model.complexOutput:
        if useLowerThres:
            self.config.model.conf_thres = old_thres
        self.writeGraph(counter, os.path.join(self.savePath, exp_path))

        # clean up model
        self.config.model.deinitialize()
        with self.threadDone.get_lock():
            self.threadDone.value = True
        self.finished.emit()

class ExperimentResultWorker(QObject):
    finished = pyqtSignal()
    finishedImage = pyqtSignal(QPixmap)
    finishedGraph = pyqtSignal(list)

    def __init__(self, imagePath, config, expName, augPosition=None, argPosition=None) -> None:
        super(ExperimentResultWorker, self).__init__()
        self.imagePath = imagePath # a single image
        self.config = config
        if self.imagePath:
            if self.config.labelType == 'coco': 
                self.imgID = self.imagePath
                _filename = self.config.labels['coco'].loadImgs(self.imagePath)[0]['file_name']
                self.imagePath = os.path.join(self.config.labels['root'], _filename)
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
        assert not _img is None

        if len(self.config.mainAug) > 0:
            if self.config.isCompound:
                for aug in self.config.mainAug:
                    _img = aug(_img, aug.args[self.argPosition])
            else:
                aug = self.config.mainAug.__pipeline__[self.augPosition]
                _img = aug(_img, aug.args[self.argPosition])
        
        # read in detection:
        _imgRoot = self.imagePath.split('/')[-1].split('.')[0]

        '''
        if self.config.labelType == 'coco':
            f = open( os.path.join(_folder_path, 'cocoRes.json') )
            _f = json.load(f)
            dets = []
            for item in _f:
                if self.imgID == item['image_id']:
                    _det = list(map(int, item['bbox']))
                    _conf = item['score']
                    _det = [_det[0], _det[1], _det[2]+_det[0], _det[3]+_det[1]]
                    if _conf > self.config.model.conf_thres:
                        _img = cv2.rectangle(_img, (_det[0],_det[1]), (_det[2],_det[3]), (0,0,255), thickness=3)
            self.finishedImage.emit(convertCV2QT(_img, 391, 231))
            self.finished.emit()
            return
        '''

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
                if d[1] > self.config.model.conf_thres: 
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

        if self.config.modelName == 'Semantic Segmentation' and len(_graphs.shape) > 1:
            _graphs = _graphs.squeeze(2)

        if self.config.isCompound:
            # segmentation specific stuff:
            if self.config.modelName == 'Semantic Segmentation':
                _graphs = _graphs.squeeze(0)
                _graphs = np.transpose(_graphs, (1,0))

            _title = ", ".join(list(graphContent.keys()))
            _items = list(graphContent.values())
            argLen = len(_items[0])
            #_g = _graphs[self.argPosition]
            _x = [i for i in range(argLen)]
            #_x = [",".join(_items[:, i]) for i in range(argLen)] # might be slow
            ax.set_title(_title)

            for _g in _graphs:
                print(_x, _g, argLen)
                ax.plot(_x, _g, 'o-')
        else:
            if self.config.modelName == 'Semantic Segmentation':
                if len(_graphs.shape) > 1:
                    _g = _graphs[self.augPosition]
                else:
                    _g = _graphs[self.augPosition]
                    _g = np.array(_g).squeeze(1)
            else:
                _g = _graphs[self.augPosition]

            _keys = list(graphContent.keys())
            _items = np.array(list(graphContent.values()))
            _title = _keys[self.augPosition]
            #_x = [i for i in range(len(_items[self.argPosition]))]
            _x = _items[self.augPosition]
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
        #self.textProgress.setEnabled(False)
        self.config = config
        self._progressMove = 1/len(self.config.imagePaths)
        self.config.expName = createExperimentName(self.config.savePath)

        # image gui controls:
        self.currentIdx = 0
        self.currentArgIdx = 0
        self.currentGraphIdx = 0
        self.currentImg = None

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
        self.previewImage.clicked.connect(lambda: self.showImage())
        #self.forwardGraph.clicked.connect(lambda: self.changeOnGraphButton(1))
        #self.backGraph.clicked.connect(lambda: self.changeOnGraphButton(-1))

        # multithreading stuff for updates after experiment:
        self.threadDone = Value('b', 0)
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
        self.worker = ExperimentWorker(self.config, self.config.savePath, self.threadDone)
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

    def showImage(self):

        # Convert the image from Pixmap to Image
        img = QImage(self.currentImg)

        img.save('test.jpg', 'jpg')

        # Read in the image with opencv and show the image
        img = cv2.imread('test.jpg')

        width = int(img.shape[1] * 5)
        height = int(img.shape[0] * 5)
        dim = (width, height)  

        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        cv2.imshow('Image', resized)

        # Delete the temp file
        os.remove('test.jpg') 

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
        self.currentImg = img
        self.previewImage.setIcon(QIcon(img))
        self.previewImage.setIconSize(QSize(500,500))

    def updateGraph(self, ax_list):
        fig, ax = ax_list
        self.graphWidget.canvas.axes.clear()

        for i in range(len(ax.lines)):
            line = ax.lines[i]
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            # self.graphWidget.canvas.axes.clear()
            self.graphWidget.canvas.axes.plot(x_data, y_data, 'o-')
        self.graphWidget.canvas.axes.set_title(ax.get_title())
        # self.graphWidget.canvas.axes.set_xlabel(str(ax.xaxis.get_label()))
        # self.graphWidget.canvas.axes.set_ylabel(str(ax.yaxis.get_label()))
        self.graphWidget.canvas.axes.set_xlabel("Augment Level")

        if self.config.labelType == 'coco' or self.config.labelType == 'voc':
            self.graphWidget.canvas.axes.set_ylabel("mAP")
        else:
            self.graphWidget.canvas.axes.set_ylabel("Accuracy")
        # self.graphWidget.canvas.axes.legend()
        self.graphWidget.canvas.draw()

    def refreshGraphResults(self,i):
        augPosition = self.augComboBox.currentIndex()
        # graphing:
        self.workerGraph = ExperimentResultWorker(None, self.config, self.config.expName, argPosition=self.currentArgIdx, augPosition=augPosition)
        self.afterGraphExpThread.started.connect(self.workerGraph.runGraph)
        self.workerGraph.finished.connect(self.afterGraphExpThread.quit)
        self.workerGraph.finished.connect(self.afterGraphExpThread.wait)
        self.workerGraph.finished.connect(self.workerGraph.deleteLater)
        self.workerGraph.finishedGraph.connect(self.updateGraph)
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

    def closeEvent(self, event):
        with self.threadDone.get_lock():
            if not self.threadDone.value:
                '''
                reply = QMessageBox.question(self, 'Question',
                    "Are you sure you want to close the application?",
                    QMessageBox.Yes,
                    QMessageBox.No)
                if reply == QMessageBox.Yes:
                    if self.thread:
                        self.thread.terminate()
                        self.thread.quit()
                    del self.thread
                    #super(ExperimentDialog, self).closeEvent(event)
                    self.closeEvent(event)
                else:
                    event.ignore()
                '''
                self.close()
            else:
                self.close()

    def _stop_thread(self):
        #self.stop_update.setVisible(False)
        #self.start_update.setVisible(True)
        self.thread.terminate()         
        self.thread = None