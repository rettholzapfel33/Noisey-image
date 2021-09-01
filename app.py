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

import cv2
from functools import partial

currPath = str(Path(__file__).parent.absolute()) + '/'
tmpPath = currPath + 'src/tmp_results/'


# Converts opencv image to qt image
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

        else:
            self.progress.emit(1)  
            CLASSES = os.path.join(currPath, 'src/obj_detector/cfg', 'coco.names')
            CFG = os.path.join(currPath, 'src/obj_detector/cfg', 'yolov3.cfg')
            WEIGHTS = os.path.join(currPath,'src/obj_detector/weights','yolov3.weights')

            self.progress.emit(2)  
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

        self.ui.progressBar.hide()

        self.ui.comboBox.addItems(["Semantic Segmentation", "Object Detection (YOLOv3)"])

        # QActions
        self.build_qactions()
        
        # self.ui.default = QtWidgets.QAction(self)
        # self.ui.default.setObjectName("default")
        # self.ui.default.setIconText("default_traffic")
        # self.ui.default.triggered.connect(lambda: self.default(self.ui.default ,"car detection.png"))

        # self.ui.default2 = QtWidgets.QAction(self)
        # self.ui.default2.setObjectName("default2")
        # self.ui.default2.setIconText("default_100_faces")
        # self.ui.default2.triggered.connect(lambda: self.default(self.ui.default2, "100FACES.jpg"))

        # self.ui.toolButton.addActions([self.ui.default, self.ui.default2])
        # self.ui.toolButton.setDefaultAction(self.ui.default)


        # Class variables
        self.noiseImg = None
        self.predictedImg = None
        self.predictedQtImg = None
        self.predictedColor = None
        self.predictedQtColor = None
        self.pred = None

        # Buttons
        self.ui.pushButton.clicked.connect(self.noise_gen)
        self.ui.pushButton_2.clicked.connect(self.start_model)
        self.ui.pushButton_4.clicked.connect(self.quitApp)

        # Menubar buttons
        self.ui.actionOpen.triggered.connect(lambda: self.open_file())
        self.ui.actionIncrease_Size.triggered.connect(self.increaseFont)
        self.ui.actionDecrease_Size.triggered.connect(self.decreaseFont)

        # Noise generator
        self.ui.horizontalSlider.valueChanged.connect(lambda: self.ui.doubleSpinBox.setValue(self.ui.horizontalSlider.value()))
        #self.ui.doubleSpinBox.valueChanged.connect(lambda: self.ui.horizontalSlider.setValue(int(self.ui.doubleSpinBox.value())))
        self.ui.doubleSpinBox.valueChanged.connect(self.noise_gen)

        self.ui.listWidget.currentItemChanged.connect(self.change_selection)

        self.default_img()
        self.ui.centralwidget.setFont(QtGui.QFont('Ubuntu', 10))


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
            action.triggered.connect(partial(self.default, action, "default_imgs/" + file))
            self.qactions.append(action)
            

        self.ui.toolButton.addActions(self.qactions)
        self.ui.toolButton.setDefaultAction(self.qactions[0])

    def default(self, qaction, fileName):
        self.default_img(fileName)
        self.ui.toolButton.setDefaultAction(qaction)


    def default_img(self, fileName = "MISC1/car detection.png"):
        print(currPath + "imgs/" + fileName)
        self.open_file(currPath + "imgs/" + fileName)
        #self.ui.original_2.setPixmap(QtGui.QPixmap(currPath+"tmp_results/pred_color.png"))
        #self.ui.preview_2.setPixmap(QtGui.QPixmap(currPath+"tmp_results/dst.png"))

        self.ui.horizontalSlider.setValue(5)
        self.noise_gen()

    def open_file(self, fileName = None):
        if(fileName == None):
            fileName = QtWidgets.QFileDialog.getOpenFileName(self, "Select image", filter="Image files (*.jpg *.png *.bmp)")
            fileName = fileName[0]
        
        img = cv2.imread(fileName)
        
        self.ui.original.addImg(img)
        self.ui.original.setPixmap(QtGui.QPixmap(fileName))
        self.noise_gen()
        

    def noise_gen(self):
        originalImg = self.ui.original.getImg()

        if(originalImg is None):
            self.ui.statusbar.showMessage("Import an image first.", 3000)
            return

        noise_level = self.ui.doubleSpinBox.value() / 100
        print("noise probability: ", noise_level)
        
        cv_img = add_noise_img(originalImg, noise_level)

        self.noiseImg = cv_img

        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qt_img = convert_cvimg_to_qimg(img_rgb)

        self.ui.preview.setPixmap(QtGui.QPixmap.fromImage(qt_img))

    def reportProgress(self, n):
        self.ui.progressBar.setValue(n)

    def change_selection(self, current):
        if(current == None):
            return

        originalImg = self.noiseImg

        #print(current.text())

        if(current.text() == "all"):
            self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(self.predictedQtColor))
            self.ui.preview_2.setPixmap(QtGui.QPixmap.fromImage(self.predictedQtImg))
        else:
            img = new_visualize_result(self.pred, originalImg, current.text())
            qImg_color = convert_cvimg_to_qimg(img[0])
            qImg_overlay = convert_cvimg_to_qimg(img[1])
            self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(qImg_color))
            self.ui.preview_2.setPixmap(QtGui.QPixmap.fromImage(qImg_overlay))

    def display_result(self, result):
        comboModelType = self.ui.comboBox.currentText()

        if comboModelType == 'Semantic Segmentation':
            self.pred = result[2]
            self.predictedImg = result[0]
            self.predictedColor = result[1]

            self.predictedQtImg = convert_cvimg_to_qimg(result[0])
            self.predictedQtColor = convert_cvimg_to_qimg(result[1])
            self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(self.predictedQtColor))
            self.ui.preview_2.setPixmap(QtGui.QPixmap.fromImage(self.predictedQtImg))
        else:
            self.pred = result[1]
            self.predictedImg = result[0]
            self.predictedQtImg = convert_cvimg_to_qimg(result[0])
            self.ui.original_2.setPixmap(QtGui.QPixmap.fromImage(self.predictedQtImg))
            self.ui.preview_2.clear()

    def display_colors(self, names):
        for x in names:
            i = QtWidgets.QListWidgetItem(x)
            i.setBackground(QtGui.QColor(names[x][0], names[x][1], names[x][2]))
            self.ui.listWidget.addItem(i)
        

    def start_model(self):
        self.ui.progressBar.show()
        self.ui.listWidget.clear()
        self.ui.original_2.clear()
        self.ui.preview_2.clear()


        self.thread = QtCore.QThread()
        self.worker = Worker()

        detectedNames = {"all": [255,255,255]}
        display_sep = self.ui.checkBox_2.isChecked()

        comboModelType = self.ui.comboBox.currentText()
        
        
        if(self.ui.original.getImg() is None):
            self.ui.statusbar.showMessage("Import an image first!", 3000)
            return
        elif(self.noiseImg is None):
            self.ui.statusbar.showMessage("Add noise to the image first!", 3000)
            return

        if comboModelType == 'Semantic Segmentation':
            self.worker.setup(self.noiseImg, tmpPath, display_sep, detectedNames, 'segmentation')
        else:
            self.worker.setup(self.noiseImg, tmpPath, display_sep, detectedNames, 'yolov3')


        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.ui.progressBar.hide)
        self.worker.finished.connect(self.display_result)
        self.worker.finished.connect(lambda: self.display_colors(detectedNames))
        self.worker.progress.connect(self.reportProgress)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    window = mainWindow()
    window.show()
    window.showMaximized()

    app.exec_()
    
