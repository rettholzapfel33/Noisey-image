# System libs
import os
from pathlib import Path
import PIL.Image

# Sementic segmentation
from src.predict_img import start_from_gui, new_visualize_result
from src.noise_image import add_noise_img

# import yolov3 stuff:
import src.obj_detector.detect as detect
from src.obj_detector.models import load_model
from src.obj_detector.utils.utils import load_classes

# PyQt5
from PyQt5 import QtCore, QtWidgets, QtGui
from src.window import Ui_MainWindow
from PyQt5.QtCore import Qt

import cv2
from functools import partial

currPath = str(Path(__file__).parent.absolute()) + '/'
tmpPath = currPath + 'src/tmp_results/'


# Converts opencv image to RGB first then to qt image
def convert_cvimg_to_qimg(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    height, width, channel = cv_img.shape
    bytesPerLine = 3 * width

    qt_img = QtGui.QImage(cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
    return qt_img


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
        if self.model_type == 'segmentation':
            result = start_from_gui(self.files, tmpPath, self.progress, self.ifDisplay)

        else:
            self.progress.emit(1)  
            CLASSES = os.path.join(currPath, 'src/obj_detector/cfg', 'coco.names')
            CFG = os.path.join(currPath, 'src/obj_detector/cfg', 'yolov3.cfg')
            WEIGHTS = os.path.join(currPath,'src/obj_detector/weights','yolov3.weights')

            self.progress.emit(2)  
            yolo = load_model(CFG, WEIGHTS)
            
             
            classes = load_classes(CLASSES)  # List of class names
            
            result = []
            for img in self.files:
                dets = detect.detect_image(yolo, img)
                np_img = detect._draw_and_return_output_image(img, dets, 416, classes)
                result.append((np_img, dets))
                self.progress.emit(3) 

            #image_dst = os.path.join(tmpPath, 'yolo_output.png')
            #cv2.imwrite(image_dst, np_img)
               
            self.progress.emit(4)   

            if (self.ifDisplay==1):
                PIL.Image.fromarray(np_img).show()

        self.finished.emit((result, self.listWidgets))


class mainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.progressBar.hide()
        self.ui.progressBar_2.hide()

        self.ui.comboBox.addItems(["Semantic Segmentation", "Object Detection (YOLOv3)"])

        # QActions
        self.build_qactions()
        self.qactions[0].trigger()

        # Buttons
        self.ui.pushButton.clicked.connect(self.noise_gen)
        self.ui.pushButton_2.clicked.connect(self.run_model)
        self.ui.pushButton_3.clicked.connect(self.noise_gen_all)
        self.ui.pushButton_4.clicked.connect(self.quitApp)
        self.ui.pushButton_5.clicked.connect(self.run_model_all)

        # Menubar buttons
        self.ui.actionOpen.triggered.connect(lambda: self.open_file())
        self.ui.actionIncrease_Size.triggered.connect(self.increaseFont)
        self.ui.actionDecrease_Size.triggered.connect(self.decreaseFont)

        # Noise generator
        self.ui.horizontalSlider.valueChanged.connect(lambda: self.ui.doubleSpinBox.setValue(self.ui.horizontalSlider.value()))
        #self.ui.doubleSpinBox.valueChanged.connect(lambda: self.ui.horizontalSlider.setValue(int(self.ui.doubleSpinBox.value())))
        self.ui.doubleSpinBox.valueChanged.connect(self.noise_gen)

        # Qlistwidget signals
        self.ui.listWidget.currentItemChanged.connect(self.change_seg_selection)
        self.ui.fileList.currentItemChanged.connect(self.change_file_selection)

        # Font
        self.ui.centralwidget.setFont(QtGui.QFont('Ubuntu', 10))

        # Drag and drop
        self.ui.original.imageDropped.connect(self.open_file)

        self.ui.fileList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.fileList.customContextMenuRequested.connect(self.listwidgetmenu)
        self.ui.fileList.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        

    def listwidgetmenu(self, position):
        rightMenu = QtWidgets.QMenu(self.ui.fileList)
        removeAction = QtWidgets.QAction("close", self, triggered = self.close)
        
        rightMenu.addAction(self.ui.actionOpen)

        if self.ui.fileList.itemAt(position):
            rightMenu.addAction(removeAction)

        rightMenu.exec_(self.ui.fileList.mapToGlobal(position))


    def close(self):
        items = self.ui.fileList.selectedItems()
        
        for item in items:
            row = self.ui.fileList.row(item)
            self.ui.fileList.takeItem(row)

    def increaseFont(self):
        self.ui.centralwidget.setFont(QtGui.QFont('Ubuntu', self.ui.centralwidget.fontInfo().pointSize() + 1))
        #print(self.ui.centralwidget.fontInfo().pointSize())

    def decreaseFont(self):
        self.ui.centralwidget.setFont(QtGui.QFont('Ubuntu', self.ui.centralwidget.fontInfo().pointSize() - 1))

    def quitApp(self):
        quit()

    def build_qactions(self):
        mypath = currPath + "imgs/default_imgs"
        onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
        self.qactions = []

        for file in onlyfiles:
            action = QtWidgets.QAction(self)
            action.setObjectName(file)
            action.setIconText("default " + file)
            action.triggered.connect(partial(self.default_qaction, action, "default_imgs/" + file))
            self.qactions.append(action)
            

        self.ui.toolButton.addActions(self.qactions)
        self.ui.toolButton.setDefaultAction(self.qactions[0])

    def default_qaction(self, qaction, fileName):
        self.open_file(currPath + "imgs/" + fileName)
        self.ui.toolButton.setDefaultAction(qaction)


    # def default_img(self, fileName = "MISC1/car detection.png"):
    #     print(currPath + "imgs/" + fileName)
    #     self.open_file(currPath + "imgs/" + fileName)
        #self.ui.original_2.setPixmap(QtGui.QPixmap(currPath+"tmp_results/pred_color.png"))
        #self.ui.preview_2.setPixmap(QtGui.QPixmap(currPath+"tmp_results/dst.png"))

        #self.ui.horizontalSlider.setValue(5)
        #self.noise_gen()

    def open_file(self, filePaths = None):
        if(filePaths == None):
            filePaths = QtWidgets.QFileDialog.getOpenFileNames(self, "Select image", filter="Image files (*.jpg *.png *.bmp)")
            filePaths = filePaths[0]
        elif(isinstance(filePaths, list) == 0):
            filePaths = [filePaths]

        new_item = None
    
        for filePath in filePaths:

            fileName = filePath[filePath.rfind('/') + 1:]

            items = self.ui.fileList.findItems(fileName, QtCore.Qt.MatchExactly)
            if(len(items) > 0):
                self.ui.statusbar.showMessage("File already opened", 3000)
                continue

            img = cv2.imread(filePath)

            new_item = QtWidgets.QListWidgetItem()
            new_item.setText(fileName)
            new_item.setData(QtCore.Qt.UserRole, {'filePath':filePath, 'img':img})
            self.ui.fileList.addItem(new_item)
            

        if(new_item is not None):
            self.ui.original.setPixmap(QtGui.QPixmap(filePath))
            self.ui.fileList.setCurrentItem(new_item)

            self.noise_gen()

            self.ui.original_2.clear()
            self.ui.preview_2.clear()


    def noise_gen(self):
        qListItem = self.ui.fileList.currentItem()
        originalImg = qListItem.data(QtCore.Qt.UserRole)['img']

        if(originalImg is None):
            self.ui.statusbar.showMessage("Import an image first.", 3000)
            return

        noise_level = self.ui.doubleSpinBox.value() / 100
        print("noise probability: ", noise_level)
        
        cv_img = add_noise_img(originalImg, noise_level)

        temp = qListItem.data(QtCore.Qt.UserRole)
        temp['noiseImg'] = cv_img
        qListItem.setData(QtCore.Qt.UserRole, temp)

        qt_img = convert_cvimg_to_qimg(cv_img)

        self.ui.preview.setPixmap(QtGui.QPixmap.fromImage(qt_img))

    def noise_gen_all(self):
        lw = self.ui.fileList
        
        items = []
        for x in range(lw.count()):
            if(lw.item(x) != lw.currentItem()):
                items.append(lw.item(x))

        noise_level = self.ui.doubleSpinBox.value() / 100

        for item in items:
            temp = item.data(QtCore.Qt.UserRole)
            cv_img = add_noise_img(temp['img'], noise_level)
            temp['noiseImg'] = cv_img
            item.setData(QtCore.Qt.UserRole, temp)

        self.noise_gen()

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
        originalImg = qListItem.data(QtCore.Qt.UserRole)['img']
        noiseImg = qListItem.data(QtCore.Qt.UserRole).get('noiseImg')
        predictedImg = qListItem.data(QtCore.Qt.UserRole).get('predictedImg')
        predictedColor = qListItem.data(QtCore.Qt.UserRole).get('predictedColor')
        items = qListItem.data(QtCore.Qt.UserRole).get('items')

        self.ui.listWidget.clear()

        originalQtImg = convert_cvimg_to_qimg(originalImg) 
        self.ui.original.setPixmap(QtGui.QPixmap.fromImage(originalQtImg))

        if(noiseImg is not None):
            noiseQtImg = convert_cvimg_to_qimg(noiseImg)
            self.ui.preview.setPixmap(QtGui.QPixmap.fromImage(noiseQtImg))
        else:
            self.ui.preview.clear()

        if(predictedImg is not None):
            predictedQtImg = convert_cvimg_to_qimg(predictedImg)
            self.ui.preview_2.setPixmap(QtGui.QPixmap.fromImage(predictedQtImg))
        else:
            self.ui.preview_2.clear()

        if(predictedColor is not None):
            predictedQtColor = convert_cvimg_to_qimg(predictedColor)
            self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(predictedQtColor))
        else:
            self.ui.original_2.clear()

        if(items is not None):
            for x in items:
                i = QtWidgets.QListWidgetItem(x)
                i.setBackground(QtGui.QColor(items[x][0], items[x][1], items[x][2]))
                self.ui.listWidget.addItem(i)

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

        #print(current.text())

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
            if comboModelType == 'Semantic Segmentation':
                temp['pred'] = result[2]
                temp['predictedImg'] = result[0]
                temp['predictedColor'] = result[1]

                if(self.ui.fileList.currentItem() == qListItem):
                    predictedQtImg = convert_cvimg_to_qimg(result[0])
                    predictedQtColor = convert_cvimg_to_qimg(result[1])
                    self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(predictedQtColor))
                    self.ui.preview_2.setPixmap(QtGui.QPixmap.fromImage(predictedQtImg))
            else:
                temp['pred'] = result[1]
                temp['predictedImg'] = result[0]
                
                if(self.ui.fileList.currentItem() == qListItem):
                    predictedQtImg = convert_cvimg_to_qimg(result[0])
                    self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(predictedQtImg))
                    self.ui.preview_2.clear()

            qListItem.setData(QtCore.Qt.UserRole, temp)

    def display_items(self, results):
        qListItems = results[1]
        seg_results = results[0]

        for i, result in enumerate(seg_results):
            qListItem = qListItems[i]
            names = result[3]
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
        img = qListItem.data(QtCore.Qt.UserRole).get('img')
        noiseImg = qListItem.data(QtCore.Qt.UserRole).get('noiseImg')

        if(img is None):
            self.ui.statusbar.showMessage("Import an image first!", 3000)
            return
        elif(noiseImg is None):
            self.ui.statusbar.showMessage("Add noise to the image first!", 3000)
            noiseImg = img

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
    