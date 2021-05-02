from pathlib import Path

from predict_img import start_from_gui, new_visualize_result
#from noise_video_gen import *
from noise_image import add_noise_img

from PyQt5 import QtCore, QtWidgets, QtGui
from window import Ui_MainWindow

from cv2 import imread

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

    def setup(self, filename, tmpPath, display, detectedNames):
        self.filename = filename
        self.tmpPath = tmpPath
        self.display = display
        self.detectedNames = detectedNames

    def run(self):
         result = start_from_gui(self.filename, self.tmpPath, self.progress, self.detectedNames, self.display)
         self.finished.emit(result)


class mainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #self.ui.stackedWidget.setCurrentWidget(self.ui.page_3)
        self.ui.progressBar.hide()

        self.ui.comboBox.addItems(["Sementic Segmentation"])

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
        img = imread(fileName[0])

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

        # cv_img[0] is the one with text
        qt_img = convert_cvimg_to_qimg(cv_img)

        self.ui.preview.setPixmap(QtGui.QPixmap.fromImage(qt_img))

    def reportProgress(self, n):
        self.ui.progressBar.setValue(n)

    def change_selection(self, current):
        if(current == None):
            return

        print(current.text())

        if(current.text() == "all"):
            self.ui.preview.setPixmap(QtGui.QPixmap.fromImage(self.predictedQtImg))
        else:
            img = new_visualize_result(self.pred, self.originalImg, current.text())
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

        if(self.ui.checkBox_3.isChecked() == True):
            self.worker.setup(self.noiseImg, tmpPath, display_sep, detectedNames)
        else:
            self.worker.setup(self.originalImg, tmpPath, display_sep, detectedNames)


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
