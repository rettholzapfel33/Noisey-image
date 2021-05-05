# System libs
import sys
import os
from pathlib import Path
import PIL.Image

# Sementic segmentation
from predict_img import start_from_gui, new_visualize_result
from noise_image import add_noise_img

# import yolov3 stuff:
import obj_detector.detect as detect
from obj_detector.models import load_model
from obj_detector.utils.utils import load_classes, rescale_boxes, non_max_suppression, to_cpu, print_environment_info

# PyQt5
from PyQt5 import QtCore, QtWidgets, QtGui
from window import Ui_MainWindow

import cv2


currPath = str(Path(__file__).parent.absolute()) + '/'
tmpPath = currPath + 'tmp_results/'


def convert_cvimg_to_qimg(cv_img):
    height, width, channel = cv_img.shape
    bytesPerLine = 3 * width

    qt_img = QtGui.QImage(cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
    return qt_img


class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal(tuple)
    progress = QtCore.pyqtSignal(int)

    def setup(self, filename, tmpPath, display, detectedNames, model_type):
        self.filename = filename
        self.tmpPath = tmpPath
        self.display = display
        self.detectedNames = detectedNames
        assert model_type == 'segmentation' or model_type == 'yolov3', "Model Type %s is not a defined term!"%(model_type)
        self.model_type = model_type

    def run(self):

        if self.model_type == 'segmentation':
            result = start_from_gui(self.filename, self.tmpPath, self.progress, self.detectedNames, self.display)
            print(result)
        else:
            self.progress.emit(1)  
            CLASSES = os.path.join(currPath, 'obj_detector/cfg', 'coco.names')
            CFG = os.path.join(currPath, 'obj_detector/cfg', 'yolov3.cfg')
            WEIGHTS = os.path.join(currPath,'obj_detector/weights','yolov3.weights')
            #self.progress.emit(2)  
            yolo = load_model(CFG, WEIGHTS)
            self.progress.emit(3)  
            classes = load_classes(CLASSES)  # List of class names
            dets = detect.detect_image(yolo, self.filename)
            np_img = detect._draw_and_return_output_image(self.filename, dets, 416, classes)
            image_dst = os.path.join(tmpPath, 'yolo_output.png')
            #cv2.imwrite(image_dst, np_img)
            result = (np_img, dets)# output an image:   
            self.progress.emit(4)   

            if (self.display==1):
                PIL.Image.fromarray(np_img).show()

        self.finished.emit(result)


class mainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #self.ui.stackedWidget.setCurrentWidget(self.ui.page_3)
        self.ui.progressBar.hide()

        self.ui.comboBox.addItems(["Semantic Segmentation", "Object Detection (YOLOv3)"])

        # Class variables
        self.originalImg = None
        self.originalImgPath = None
        self.noiseImg = None
        self.predictedImg = None
        self.predictedQtImg = None
        self.pred = None

        # Buttons
        self.ui.actionOpen.triggered.connect(self.file_browse)
        #self.ui.pushButton_browse_file.clicked.connect(lambda : self.file_browse(self.ui.lineEdit_filename))
        #self.ui.pushButton_browse_file_2.clicked.connect(lambda: self.file_browse(self.ui.lineEdit_filename_2))
        self.ui.pushButton.clicked.connect(self.noise_gen)
        self.ui.pushButton_2.clicked.connect(self.start_model)
        
        # Changing pages
        # self.ui.pb_noise_gen.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_2))
        # self.ui.pb_sementic_seg.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page))
        # self.ui.pb_back.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_3))
        # self.ui.pb_back_2.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_3))

        self.ui.checkBox.stateChanged.connect(self.realTimePreview)

        self.ui.horizontalSlider.valueChanged.connect(lambda: self.ui.label_7.setText(str(self.ui.horizontalSlider.value() / 100)))

        self.ui.listWidget.currentItemChanged.connect(self.change_selection)


    def file_browse(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, "Select image", filter="Image files (*.jpg *.png)")
       
        #lineEdit.setText(fileName[0])
        #print(fileName[0])
        img = cv2.imread(fileName[0])

        self.originalImgPath = fileName[0]
        self.originalImg = img
        self.ui.original.setPixmap(QtGui.QPixmap(self.originalImgPath))
        
    def realTimePreview(self):
        if(self.ui.checkBox.isChecked() == True):
            self.ui.horizontalSlider.valueChanged.connect(self.noise_gen)
        else:
            self.ui.horizontalSlider.valueChanged.disconnect(self.noise_gen)

    def noise_gen(self):

        if(self.originalImg is None):
            self.ui.statusbar.showMessage("Import an image first.", 3000)
            return

        noise_level = self.ui.horizontalSlider.value() / 100
        # out = self.ui.lineEdit.text() + ".jpg"
        # out = tmpPath + out
        # print(out)
        cv_img = add_noise_img(self.originalImg, noise_level)

        self.noiseImg = cv_img

        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # cv_img[0] is the one with text
        qt_img = convert_cvimg_to_qimg(img_rgb)

        self.ui.preview.setPixmap(QtGui.QPixmap.fromImage(qt_img))

    def reportProgress(self, n):
        self.ui.progressBar.setValue(n)

    def change_selection(self, current):
        if(current == None):
            return

        if(self.ui.checkBox_3.isChecked() == True):
            originalImg = self.noiseImg
        else:
            originalImg = self.originalImg

        #print(current.text())

        if(current.text() == "all"):
            self.ui.preview.setPixmap(QtGui.QPixmap.fromImage(self.predictedQtImg))
        else:
            img = new_visualize_result(self.pred, originalImg, current.text())
            qImg = convert_cvimg_to_qimg(img)
            self.ui.preview.setPixmap(QtGui.QPixmap.fromImage(qImg))

    def display_result(self, result):
        self.pred = result[1]
        self.predictedImg = result[0]

        self.predictedQtImg = convert_cvimg_to_qimg(result[0])
        self.ui.preview.setPixmap(QtGui.QPixmap.fromImage(self.predictedQtImg))

    def start_model(self):
        self.ui.progressBar.show()
        self.ui.listWidget.clear()
        self.ui.preview.clear()

        #if(self.ui.checkBox_3.isChecked == True):

        self.thread = QtCore.QThread()
        self.worker = Worker()

        detectedNames = ["all"]
        display_sep = self.ui.checkBox_2.isChecked()

        comboModelType = self.ui.comboBox.currentText()
        if(self.ui.checkBox_3.isChecked() == True and self.noiseImg is not None):
            if comboModelType == 'Semantic Segmentation':
                self.worker.setup(self.noiseImg, tmpPath, display_sep, detectedNames, 'segmentation')
            else:
                self.worker.setup(self.noiseImg, tmpPath, display_sep, detectedNames, 'yolov3')

        elif(self.ui.checkBox_3.isChecked() == True and self.noiseImg is None):
            self.ui.statusbar.showMessage("Add noise to the image first!", 3000)
            return

        elif(self.ui.checkBox_3.isChecked() == False and self.originalImg is None):
            self.ui.statusbar.showMessage("Import an image first!", 3000)
            return
            
        else:
            if comboModelType == 'Semantic Segmentation':
                self.worker.setup(self.originalImg, tmpPath, display_sep, detectedNames, 'segmentation')
            else:
                self.worker.setup(self.originalImg, tmpPath, display_sep, detectedNames, 'yolov3')

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.ui.progressBar.hide)
        #self.worker.finished.connect(lambda: self.ui.preview.setPixmap(QtGui.QPixmap(tmpPath + 'dst.png')))
        self.worker.finished.connect(self.display_result)
        self.worker.finished.connect(lambda: self.ui.listWidget.addItems(detectedNames))
        self.worker.progress.connect(self.reportProgress)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()

        #start_from_gui(self.ui.lineEdit_filename_2.text(), tmpPath, display=display_sep)



if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    window = mainWindow()
    window.show()
    
    app.exec_()
