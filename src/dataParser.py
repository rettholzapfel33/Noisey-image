# System libs
import os
import yaml
import json
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt
from multiprocessing import Queue
import time

# PyQt5
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QListWidgetItem, QMessageBox, QWidget
from PyQt5 import uic

# import utilities:
from src.yamlDialog import Ui_Dialog

# eval:
from src.evaluators.map_metric.lib.BoundingBox import BoundingBox

# COCO:
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class ReadYAMLProgressWindow(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi('./src/qt_designer_file/yamlProgress.ui', self)

class yamlWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    logProgress = pyqtSignal(str)

    def __init__(self, queue:Queue) -> None:
        super(yamlWorker, self).__init__()
        self.filePath = ""
        self.queue = queue
        
    def run(self):
        time.sleep(0.5)
        filePaths = []
        label_eval = 'voc' # default label type

        # Parse user-created YAML file to dataset
        with open(self.filePath) as file:
            documents = yaml.full_load(file)

        # Track what needs to be trained, validated, and tested
        trainVT = []        
        if("images" in documents):
            trainVT.append("images")
            
        checkedItems = trainVT

        # Adds file paths to files within folders specified
        for x in checkedItems:
            if(isinstance(documents[x], list)):
                filePaths.extend(documents[x])
            else:
                filePaths.append(documents[x])

        # Assign root path to dataset specified
        if "root" in documents:
            root = documents["root"]
        else: 
            root = self.filePath[:self.filePath.rfind('/') + 1]

        # Append root to include specific path
        if "path" in documents:
            root = os.path.join(root, documents["path"])

        filePaths = list(map(lambda path: root + path, filePaths))

        # Stores path to files stored in directories
        for file in filePaths:
            if(os.path.isdir(file)):
                onlyfiles = [f for f in os.listdir(file) if os.path.isfile(os.path.join(file, f))]
                onlyfiles = list(map(lambda path: os.path.join(file, path), onlyfiles))

                filePaths.remove(file)
                filePaths.extend(onlyfiles)

        # Parses label files according to dataset type (currently accepts .txt, .xml) -> Future: .json
        if "labels" in documents:
            labels_folder = os.path.join(root, documents["labels"])
            onlylabels = [f for f in os.listdir(labels_folder) if os.path.isfile(os.path.join(labels_folder, f))]
            labels = list(map(lambda path: os.path.join(labels_folder, path), onlylabels))
            labels_dic = {}

            # Parses .xml annotation files and stores in dictionary as the following:
            # { filename: [width, height, [objects]] }
            # For VOC datasets
            if documents["type"] == "voc":
                for label in labels:
                    file_content = []
                    with open(label) as f:
                        tree_root = ET.parse(f).getroot()
       
                    for x in tree_root.findall("object"):
                        base_name = tree_root[1].text
                        obj_class = [x[i].text for i in range(4)]
                        coords = [j[i].text for j in x.findall("bndbox") for i in range(len(j))]
                        w = int(coords[2])-int(coords[0])
                        h = int(coords[3])-int(coords[1])
                        box = BoundingBox(base_name, obj_class[0], coords[0], coords[1], str(w), str(h))
                        file_content.append(box)

                    labels_dic[base_name] = file_content
                labels_content = labels_dic
            # Parses .json annotation files and stores in dictionary -> for COCO datasets
            elif documents["type"] == "coco":
                # Load COCO dataset
                dataDir = root
                dataType = documents["images"]
                annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

                # Intialize COCO api for instance annotations
                labels_dic['coco'] = COCO(annFile)
                labels_dic['root'] = os.path.join(root, dataType)
                label_eval = 'coco'
                labels_content = labels_dic

            # Parses .txt annotation files
            else:
                for label in labels:
                    base=os.path.basename(label)
                    base_name = os.path.splitext(base)[0]
                    file_content = []
                    with open(label) as f:
                        for line in f:
                            _list = line.split(',')
                            if type(_list) == list:
                                _list = list(map(float, _list))
                                #for line in _list:
                                box = BoundingBox(base_name, _list[0], _list[1], _list[2], _list[3], _list[4])
                                #box = BoundingBox(base_name, "face", _list[1], _list[2], _list[3], _list[4])
                                file_content.append(box)
                            #file_content.append(_list)

                    #print(file_content)
                    labels_dic[base_name] = file_content
                label_eval = 'voc'
                labels_content = labels_dic
        self.queue.put((filePaths, (labels_content, label_eval)))
        self.finished.emit()

def read_yaml(self, filePath):
    filePaths = []
    label_eval = 'voc' # default label type

    # Parse user-created YAML file to dataset
    with open(filePath) as file:
        documents = yaml.full_load(file)

    #readWin = ReadYAMLProgressWindow()
    #readWin.show()

    # Track what needs to be trained, validated, and tested
    trainVT = []        
    if("images" in documents):
        trainVT.append("images")

    '''
    if(len(trainVT) > 1):
        dialogUI = Ui_Dialog()
        dialog = QtWidgets.QDialog()
        dialogUI.setupUi(dialog)
        for x in trainVT:
            item = QtWidgets.QListWidgetItem()
            item.setText(x)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            dialogUI.listWidget.addItem(item)
        dialog.exec_()
        if(dialog.result() == 0):
            return []
        checkedItems = []
        for index in range(dialogUI.listWidget.count()):
            if dialogUI.listWidget.item(index).checkState() == Qt.Checked:
                checkedItems.append(dialogUI.listWidget.item(index).text())
    else:
    '''
    checkedItems = trainVT

    # Adds file paths to files within folders specified
    for x in checkedItems:
        if(isinstance(documents[x], list)):
            filePaths.extend(documents[x])
        else:
            filePaths.append(documents[x])

    # Assign root path to dataset specified
    if "root" in documents:
        root = documents["root"]
    else: 
        root = filePath[:filePath.rfind('/') + 1]

    # Append root to include specific path
    if "path" in documents:
        root = os.path.join(root, documents["path"])

    filePaths = list(map(lambda path: root + path, filePaths))

    # Stores path to files stored in directories
    for file in filePaths:
        if(os.path.isdir(file)):
            onlyfiles = [f for f in os.listdir(file) if os.path.isfile(os.path.join(file, f))]
            onlyfiles = list(map(lambda path: os.path.join(file, path), onlyfiles))

            filePaths.remove(file)
            filePaths.extend(onlyfiles)

    # Parses label files according to dataset type (currently accepts .txt, .xml) -> Future: .json
    if "labels" in documents:
        labels_folder = os.path.join(root, documents["labels"])
        onlylabels = [f for f in os.listdir(labels_folder) if os.path.isfile(os.path.join(labels_folder, f))]
        labels = list(map(lambda path: os.path.join(labels_folder, path), onlylabels))
        labels_dic = {}

        # Parses .xml annotation files and stores in dictionary as the following:
        # { filename: [width, height, [objects]] }
        # For VOC datasets
        if documents["type"] == "voc":
            for label in labels:
                file_content = []
                with open(label) as f:
                    tree_root = ET.parse(f).getroot()
                
                objects = []
                for x in tree_root.findall("object"):
                    base_name = tree_root[1].text
                    obj_class = [x[i].text for i in range(4)]
                    coords = [j[i].text for j in x.findall("bndbox") for i in range(len(j))]
                    w = int(coords[2])-int(coords[0])
                    h = int(coords[3])-int(coords[1])
                    box = BoundingBox(base_name, obj_class[0], coords[0], coords[1], str(w), str(h))
                    file_content.append(box)

                labels_dic[base_name] = file_content
            labels_content = labels_dic
        # Parses .json annotation files and stores in dictionary -> for COCO datasets
        elif documents["type"] == "coco":
            # Load COCO dataset
            dataDir = root
            dataType = documents["images"]
            annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

            # Intialize COCO api for instance annotations
            labels_dic['coco'] = COCO(annFile)
            labels_dic['root'] = os.path.join(root, dataType)
            label_eval = 'coco'
            labels_content = labels_dic

        # Parses .txt annotation files
        else:
            for label in labels:
                base=os.path.basename(label)
                base_name = os.path.splitext(base)[0]
                file_content = []
                with open(label) as f:
                    for line in f:
                        _list = line.split(',')
                        if type(_list) == list:
                            _list = list(map(float, _list))
                            #for line in _list:
                            box = BoundingBox(base_name, _list[0], _list[1], _list[2], _list[3], _list[4])
                            #box = BoundingBox(base_name, "face", _list[1], _list[2], _list[3], _list[4])
                            file_content.append(box)
                        #file_content.append(_list)

                #print(file_content)
                labels_dic[base_name] = file_content
            label_eval = 'voc'
            labels_content = labels_dic
        
        self.labels = labels_dic

    return filePaths, (labels_content, label_eval)