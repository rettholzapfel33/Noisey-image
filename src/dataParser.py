# System libs
import os
import yaml

# PyQt5
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

# import utilities:
from src.yamlDialog import Ui_Dialog


def read_yaml(self, filePath):
    #print(filePath[:filePath.rfind('/') + 1])
    filePaths = []
    with open(filePath) as file:
        documents = yaml.full_load(file)
        #print(documents)

    trainVT = []
    if("train" in documents):
        trainVT.append("train")
    if("val" in documents):
        trainVT.append("val")
    if("test" in documents):
        trainVT.append("test")

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

    for x in checkedItems:
        if(isinstance(documents[x], list)):
            filePaths.extend(documents[x])
        else:
            filePaths.append(documents[x])

    root = filePath[:filePath.rfind('/') + 1]

    if "path" in documents:
        root = os.path.join(root, documents["path"])

    filePaths = list(map(lambda path: root + path, filePaths))

    for file in filePaths:
        if(os.path.isdir(file)):
            onlyfiles = [f for f in os.listdir(file) if os.path.isfile(os.path.join(file, f))]
            onlyfiles = list(map(lambda path: os.path.join(file, path), onlyfiles))
    
            filePaths.remove(file)
            filePaths.extend(onlyfiles)

    if "labels" in documents:
        labels_folder = os.path.join(root, documents["labels"])
        onlylabels = [f for f in os.listdir(labels_folder) if os.path.isfile(os.path.join(labels_folder, f))]
        labels = list(map(lambda path: os.path.join(labels_folder, path), onlylabels))

        labels_dic = {}
        for label in labels:
            file_content = []
            with open(label) as f:
                for line in f:
                    _list = line.split()
                    if type(_list) == list:
                        _list = list(map(float, _list))
                    file_content.append(_list)
            #print(file_content)
            base=os.path.basename(label)
            labels_dic[os.path.splitext(base)[0]] = file_content

        self.labels = labels_dic
    
    return filePaths
