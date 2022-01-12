# System libs
import os
from pathlib import Path
import PIL.Image
import numpy as np

# Sementic segmentation
from src.predict_img import new_visualize_result
#from src.noise_image import add_noise_img

# PyQt5
from PyQt5 import QtCore, QtWidgets, QtGui
from src.window import Ui_MainWindow
from PyQt5.QtCore import Qt
from src.yamlDialog import Ui_Dialog

import cv2
from functools import partial
import yaml

# import utilities:
from src.utils.images import convert_cvimg_to_qimg
from src.transforms import AugDialog, AugmentationPipeline, Augmentation, mainAug
from src import models
from src.utils.qt5extra import CheckState
from src.utils.weights import Downloader

CURRENT_PATH = str(Path(__file__).parent.absolute()) + '/'
TEMP_PATH = CURRENT_PATH + 'src/tmp_results/'

class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal(tuple)
    progress = QtCore.pyqtSignal(int)

    def setup(self, files, ifDisplay, model_type, listWidgets):
        self.files = files
        self.ifDisplay = ifDisplay
        self.listWidgets = listWidgets
        assert model_type == 'segmentation' or model_type == 'yolov3', "Model Type %s is not a defined term!"%(model_type)
        self.model_type = model_type

    def run(self):
        model = models._registry[self.model_type]
        self.progress.emit(1)

        model.initialize()
        self.progress.emit(2)

        result = []
        for img in self.files:
            pred = model.run(img)
            temp = model.draw(pred, img)
            temp["pred"] = pred
            result.append(temp)
            self.progress.emit(3)

        self.progress.emit(4)
        model.deinitialize()

        self.finished.emit((result, self.listWidgets))


class mainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.addWindow = AugDialog(self.ui.listAugs)
        self.addWindow.setModal(True)
        self.addWindow.demoAug()

        # Check status of configurations:
        weight_dict = {'mit_semseg':"ade20k-hrnetv2-c1", 'yolov3':"yolov3.weights"}
        self.labels = None

        if Downloader.check(weight_dict):
            self.downloadDialog = Downloader(weight_dict)
            self.downloadDialog.setModal(True)
            self.downloadDialog.show()

        self.ui.listAugs.setMaximumSize(400,100) # quickfix for sizing issue with layouts
        self.ui.deleteListAug.setMaximumWidth(30)
        self.ui.upListAug.setMaximumWidth(30)
        self.ui.downListAug.setMaximumWidth(30)

        self.ui.progressBar.hide()
        self.ui.progressBar_2.hide()

        self.ui.comboBox.addItems(["Semantic Segmentation", "Object Detection (YOLOv3)"])

        # QActions
        # Default values (images, noise, etc.) are set up here:
        self.default_img()

        # Buttons
        self.ui.pushButton.clicked.connect(self.run_model)
        #self.ui.pushButton_2.clicked.connect(self.run_model)
        #self.ui.pushButton_3.clicked.connect(self.noise_gen_all) # replace with new function
        self.ui.pushButton_4.clicked.connect(self.quitApp)
        #self.ui.pushButton_5.clicked.connect(self.run_model_all)

        # Augmentation Generator:
        self.ui.addAug.clicked.connect(self.addWindow.show)
        self.ui.demoAug.clicked.connect(self.addWindow.demoAug)
        self.ui.loadAug.clicked.connect(self.addWindow.__loadFileDialog__)
        self.ui.saveAug.clicked.connect(self.addWindow.__saveFileDialog__)
        self.ui.deleteListAug.clicked.connect(self.addWindow.__deleteItem__)
        self.ui.downListAug.clicked.connect(self.addWindow.__moveDown__)
        self.ui.upListAug.clicked.connect(self.addWindow.__moveUp__)
        self.ui.listAugs.itemChanged.connect(self.changePreviewImage)
        # access model of listwidget to detect changes
        self.addWindow.pipelineChanged.connect(self.changePreviewImage)
        #self.ui.runOnAug.stateChanged.connect(self.runAugOnImage)

        # Menubar buttons
        self.ui.actionOpen.triggered.connect(lambda: self.open_file())
        self.ui.actionIncrease_Size.triggered.connect(self.increaseFont)
        self.ui.actionDecrease_Size.triggered.connect(self.decreaseFont)

        # Qlistwidget signals
        self.ui.listWidget.currentItemChanged.connect(self.change_seg_selection)
        self.ui.fileList.currentItemChanged.connect(self.change_file_selection)

        # Font
        font = self.font()
        font.setPointSize(10)
        self.ui.centralwidget.setFont(font)

        # Drag and drop
        self.ui.original.imageDropped.connect(self.open_file)

        self.ui.fileList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.fileList.customContextMenuRequested.connect(self.listwidgetmenu)
        self.ui.fileList.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self.ui.listAugs.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

    def listwidgetmenu(self, position):
        """menu for right clicking in the file list widget"""
        right_menu = QtWidgets.QMenu(self.ui.fileList)
        remove_action = QtWidgets.QAction("close", self, triggered = self.close)

        right_menu.addAction(self.ui.actionOpen)

        if self.ui.fileList.itemAt(position):
            right_menu.addAction(remove_action)

        right_menu.exec_(self.ui.fileList.mapToGlobal(position))


    def close(self):
        """Remoes a file from the file list widget"""
        items = self.ui.fileList.selectedItems()

        for item in items:
            row = self.ui.fileList.row(item)
            self.ui.fileList.takeItem(row)

    def increaseFont(self):
        """Increses the size of font across the whole application"""
        self.ui.centralwidget.setFont(QtGui.QFont('Ubuntu', self.ui.centralwidget.fontInfo().pointSize() + 1))
        #print(self.ui.centralwidget.fontInfo().pointSize())

    def decreaseFont(self):
        """Decreses the size of font across the whole application"""
        self.ui.centralwidget.setFont(QtGui.QFont('Ubuntu', self.ui.centralwidget.fontInfo().pointSize() - 1))

    def quitApp(self):
        quit()

    def runAugOnImage(self, state):
        if state == CheckState.Checked:
            pass
        elif state == CheckState.Unchecked:
            pass


    def updateNoisePixMap(self, mat, augs):
        for aug in augs:
            mat = aug(mat, example=True)
        qt_img = convert_cvimg_to_qimg(mat)
        self.ui.preview.setPixmap(QtGui.QPixmap.fromImage(qt_img))

        return mat


    def changePreviewImage(self, *kwargs):
        #print(kwargs)
        print("recreating noisey image")
        current_item = self.ui.fileList.currentItem()
        image = cv2.imread(current_item.data(QtCore.Qt.UserRole)['filePath'])
        #if image is not None:
        self.updateNoisePixMap(image, mainAug)


    def default_img(self, fileName = "MISC1/car detection.png"):
        #print(CURRENT_PATH + "imgs/" + fileName)
        self.open_file(CURRENT_PATH + "imgs/" + fileName)

        self.changePreviewImage()


    def open_file(self, filePaths = None):
        if(filePaths == None):
            filePaths = QtWidgets.QFileDialog.getOpenFileNames(self, "Select image", filter="Image files (*.jpg *.png *.bmp *.yaml)")
            filePaths = filePaths[0]
        elif(isinstance(filePaths, list) == 0):
            filePaths = [filePaths]

        new_item = None

        for filePath in filePaths:
            fileName = os.path.basename(filePath)
            items = self.ui.fileList.findItems(fileName, QtCore.Qt.MatchExactly)
            if len(items) > 0:
                self.ui.statusbar.showMessage("File already opened", 3000)
                continue

            if filePath.endswith(".yaml"):
                # return_value = self.read_yaml(filePath)
                # if(len(return_value) > 1 and type(return_value[1]) is dict):
                #     filePaths.extend(return_value[0])
                #     labels = return_value[1]
                # else:
                #     filePaths.extend(return_value)
                filePaths.extend(self.read_yaml(filePath))
                continue

            new_item = QtWidgets.QListWidgetItem()
            new_item.setText(fileName)
            new_item.setData(QtCore.Qt.UserRole, {'filePath':filePath})
            self.ui.fileList.addItem(new_item)


        if(new_item is not None):
            self.ui.original.setPixmap(QtGui.QPixmap(filePath))
            self.ui.fileList.setCurrentItem(new_item)
            self.ui.original_2.clear()
            self.ui.preview_2.clear()

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

        #print(filePaths)

        if "labels" in documents:
            labels_folder = os.path.join(root, documents["labels"])
            onlylabels = [f for f in os.listdir(labels_folder) if os.path.isfile(os.path.join(labels_folder, f))]
            labels = list(map(lambda path: os.path.join(labels_folder, path), onlylabels))

            labels_dic = {}
            for label in labels:
                file_content = []
                with open(label) as f:
                    for line in f:
                        file_content.append(line.split())
                #print(file_content)
                base=os.path.basename(label)
                labels_dic[os.path.splitext(base)[0]] = file_content

            self.labels = labels_dic
        
        return filePaths


    def noise_gen_all(self):
        lw = self.ui.fileList

        items = []
        for x in range(lw.count()):
            if(lw.item(x) != lw.currentItem()):
                items.append(lw.item(x))

        

        for item in items:
            _data = item.data(QtCore.Qt.UserRole)
            mat = np.copy(_data['img'])
            for aug in mainAug:
                mat = aug(mat, example=True)

            _data['noiseImg'] = mat
            item.setData(QtCore.Qt.UserRole, _data)

        self.changePreviewImage()

    def run_model_all(self):
        lw = self.ui.fileList

        items = []
        for x in range(lw.count()):
            #if(lw.item(x) != lw.currentItem()):
            items.append(lw.item(x))

        imgs = []
        qListItems = []

        for qListItem in items:
            img = qListItem.data(QtCore.Qt.UserRole).get('img')
            noiseImg = qListItem.data(QtCore.Qt.UserRole).get('noiseImg')

            if(img is None):
                self.ui.statusbar.showMessage("Import an image first!", 3000)
                return
            elif(noiseImg is None):
                # self.ui.statusbar.showMessage("Add noise to the image first!", 3000)
                # return
                noiseImg = img

            imgs.append(noiseImg)
            qListItems.append(qListItem)

        self.ui.pushButton_5.setEnabled(False)

        self.ui.progressBar_2.show()
        self.ui.progressBar_2.reset()
        self.ui.listWidget.clear()
        self.ui.original_2.clear()
        self.ui.preview_2.clear()

        self.ui.progressBar_2.setMaximum(len(imgs))

        self.thread = QtCore.QThread()
        self.worker = Worker()

        display_sep = False
        comboModelType = self.ui.comboBox.currentText()

        if comboModelType == 'Semantic Segmentation':
            self.worker.setup(imgs, display_sep, 'segmentation', qListItems)
        else:
            self.worker.setup(imgs, display_sep, 'yolov3', qListItems)


        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.ui.progressBar_2.hide)
        self.worker.finished.connect(self.display_result)
        if(comboModelType == "Semantic Segmentation"):
            self.worker.finished.connect(self.display_items)
        self.worker.finished.connect(lambda: self.ui.pushButton_5.setEnabled(True))
        self.worker.progress.connect(self.reportProgress2)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def reportProgress2(self, n):
        if(n == 3):
            self.ui.progressBar_2.setValue(self.ui.progressBar_2.value() + 1)

    def reportProgress(self, n):
        self.ui.progressBar.setValue(n)

    def change_file_selection(self, qListItem):
        originalImg = cv2.imread(qListItem.data(QtCore.Qt.UserRole)['filePath'])

        self.ui.listWidget.clear()

        originalQtImg = convert_cvimg_to_qimg(originalImg)
        self.ui.original.setPixmap(QtGui.QPixmap.fromImage(originalQtImg))

        # if(noiseImg is not None):
        #     noiseQtImg = convert_cvimg_to_qimg(noiseImg)
        #     self.ui.preview.setPixmap(QtGui.QPixmap.fromImage(noiseQtImg))
        # else:
        #     self.ui.preview.clear()
        self.changePreviewImage()


    def change_seg_selection(self, current):
        if(current == None):
            return

        qListItem = self.ui.fileList.currentItem()
        originalImg = qListItem.data(QtCore.Qt.UserRole)['img']

        predictedImg = qListItem.data(QtCore.Qt.UserRole).get('predictedImg')
        
        if(predictedImg is None):
            return

        predictedQtImg = convert_cvimg_to_qimg(predictedImg)
        predictedQtColor = convert_cvimg_to_qimg(qListItem.data(QtCore.Qt.UserRole)['predictedColor'])

        if(current.text() == "all"):
            self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(predictedQtColor))
            self.ui.preview_2.setPixmap(QtGui.QPixmap.fromImage(predictedQtImg))
        else:
            img = new_visualize_result(qListItem.data(QtCore.Qt.UserRole)['pred'], originalImg, current.text())
            qImg_color = convert_cvimg_to_qimg(img[0])
            qImg_overlay = convert_cvimg_to_qimg(img[1])
            self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(qImg_color))
            self.ui.preview_2.setPixmap(QtGui.QPixmap.fromImage(qImg_overlay))

    def display_result(self, result):
        comboModelType = self.ui.comboBox.currentText()
        qListItems = result[1]
        model_results = result[0]

        for i, result in enumerate(model_results):
            qListItem = qListItems[i]
            temp = qListItem.data(QtCore.Qt.UserRole)

            temp['pred'] = result["pred"]
            temp['predictedImg'] = result["dst"]

            if "segmentation" in result:
                temp['predictedColor'] = result["segmentation"]

                if(self.ui.fileList.currentItem() == qListItem):
                    predictedQtColor = convert_cvimg_to_qimg(result["segmentation"])
                    self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(predictedQtColor))
            else:
                self.ui.preview_2.clear()

            predictedQtImg = convert_cvimg_to_qimg(result["dst"])
            self.ui.preview_2.setPixmap(QtGui.QPixmap.fromImage(predictedQtImg))
            qListItem.setData(QtCore.Qt.UserRole, temp)

    def display_items(self, results):
        qListItems = results[1]
        seg_results = results[0]

        for i, result in enumerate(seg_results):
            qListItem = qListItems[i]
            names = result["listOfNames"]
            if(self.ui.fileList.currentItem() == qListItem):
                for x in names:
                    i = QtWidgets.QListWidgetItem(x)
                    i.setBackground(QtGui.QColor(names[x][0], names[x][1], names[x][2]))
                    self.ui.listWidget.addItem(i)

            temp = qListItem.data(QtCore.Qt.UserRole)
            temp['items'] = names
            qListItem.setData(QtCore.Qt.UserRole, temp)

        
    def run_model(self):
        qListItem = self.ui.fileList.currentItem()
        img = cv2.imread(qListItem.data(QtCore.Qt.UserRole).get('filePath'))

        if img is None:
            self.ui.statusbar.showMessage("Import an image first!", 3000)
            return

        noiseImg = self.updateNoisePixMap(img, mainAug)

        self.ui.pushButton_2.setEnabled(False)
        self.ui.progressBar.show()
        self.ui.listWidget.clear()
        self.ui.original_2.clear()
        self.ui.preview_2.clear()

        self.thread = QtCore.QThread()
        self.worker = Worker()

        #detectedNames = {"all": [255,255,255]}
        display_sep = self.ui.checkBox_2.isChecked()
        comboModelType = self.ui.comboBox.currentText()

        if comboModelType == 'Semantic Segmentation':
            self.worker.setup([noiseImg], display_sep, 'segmentation', [qListItem])
        else:
            self.worker.setup([noiseImg], display_sep, 'yolov3', [qListItem])
        
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.ui.progressBar.hide)
        self.worker.finished.connect(self.display_result)
        if(comboModelType == "Semantic Segmentation"):
            self.worker.finished.connect(self.display_items)
        self.worker.finished.connect(lambda: self.ui.pushButton_2.setEnabled(True))
        self.worker.progress.connect(self.reportProgress)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    window = mainWindow()
    window.show()
    window.showMaximized()

    app.exec_()
