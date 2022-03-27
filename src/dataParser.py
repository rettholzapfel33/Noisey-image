# System libs
import os
import yaml
import xml.etree.ElementTree as ET

# PyQt5
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

# import utilities:
from src.yamlDialog import Ui_Dialog

def read_yaml(self, filePath):
    filePaths = []

    # Parse user-created YAML file to dataset
    with open(filePath) as file:
        documents = yaml.full_load(file)

    # Track what needs to be trained, validated, and tested
    trainVT = []        
    if("train" in documents):
        trainVT.append("train")
    if("val" in documents):
        trainVT.append("val")
    if("test" in documents):
        trainVT.append("test")
    
    # print(f"TrainVT: {trainVT}")

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

        if documents["type"] == "voc":
            # for label in labels:
            #     file_content = []
            with open(labels[0]) as f:
                tree_root = ET.parse(f).getroot()
                
            # for child in tree_root:
            #     print(f"Children: {child}")
            #     for x in tree_root.findall(child.tag+"/*"):
            #         print(f"{x.tag}: {x.text}")
            # for obj in tree_root.findall("object"):
            #     print(f"Object Name: {obj[0].text}")

        else:
            # Works for files that don't need special treatment (like .txt)
            for label in labels:
                file_content = []
                with open(label) as f:
                    for line in f:
                        _list = line.split()
                        if type(_list) == list:
                            _list = list(map(float, _list))
                        file_content.append(_list)
                base=os.path.basename(label)
                labels_dic[os.path.splitext(base)[0]] = file_content
            self.labels = labels_dic
    
    return filePaths