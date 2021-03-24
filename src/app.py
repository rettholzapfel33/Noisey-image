import os, sys
from pathlib import Path

from predict_img import start_from_gui
#from noise_video_gen import *
from noise_image import add_noise_img

from PyQt5 import QtCore, QtWidgets, QtGui
from window import Ui_MainWindow

currPath = str(Path(__file__).parent.absolute()) + '/'
tmpPath = currPath + 'tmp_results/'

class mainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.stackedWidget.setCurrentWidget(self.ui.page_3)

        self.ui.pushButton_browse_file.clicked.connect(lambda : self.file_browse(self.ui.lineEdit_filename))
        self.ui.pushButton_browse_file_2.clicked.connect(lambda: self.file_browse(self.ui.lineEdit_filename_2))
        self.ui.pushButton.clicked.connect(self.noise_gen)
        self.ui.pushButton_2.clicked.connect(self.start_model)
        
        self.ui.pb_noise_gen.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_2))
        self.ui.pb_sementic_seg.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page))
        self.ui.pb_back.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_3))
        self.ui.pb_back_2.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_3))

        self.ui.checkBox.stateChanged.connect(lambda: self.ui.lineEdit_filename_2.setText(tmpPath + self.ui.lineEdit.text() + ".jpg"))

        self.ui.horizontalSlider.valueChanged.connect(lambda: self.ui.label_7.setText(str(self.ui.horizontalSlider.value() / 100)))
        self.ui.horizontalSlider.valueChanged.connect(self.noise_gen)

    def file_browse(self, lineEdit):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, "Select image", filter="Image files (*.jpg *.png)")
       
        lineEdit.setText(fileName[0])
        #print(fileName[0])
        
        if(lineEdit == self.ui.lineEdit_filename):
            self.ui.original.setPixmap(QtGui.QPixmap(lineEdit.text()))
        elif(lineEdit == self.ui.lineEdit_filename_2):
            self.ui.original2.setPixmap(QtGui.QPixmap(lineEdit.text()))

    def noise_gen(self):
        
        img = self.ui.lineEdit_filename.text()

        if(img == ""):
            return

        noise_level = self.ui.horizontalSlider.value() / 100
        out = self.ui.lineEdit.text() + ".jpg"
        out = tmpPath + out
        print(out)
        add_noise_img(img, noise_level, out)
        self.ui.preview.setPixmap(QtGui.QPixmap(out))

    def start_model(self):
        self.ui.original2.setPixmap(QtGui.QPixmap(self.ui.lineEdit_filename_2.text()))
        display_sep = self.ui.checkBox_2.isChecked()
        start_from_gui(self.ui.lineEdit_filename_2.text(), tmpPath, display=display_sep)
        self.ui.segmented.setPixmap(QtGui.QPixmap(tmpPath + 'dst.png'))

if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    window = mainWindow()
    window.show()

    app.exec_()
