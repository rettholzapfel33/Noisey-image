import sys

sys.path.append('/home/rus/Desktop/UTK/cs493/Noisey-image')
sys.path.append('/home/rus/Desktop/UTK/cs493/semantic-segmentation-pytorch')
from noise_video_gen import *
from predict_img import *
from PyQt5 import QtCore, QtWidgets

from mainwindow_ui import Ui_MainWindow


class mainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton_browse_file.clicked.connect(self.file_browse)
        self.ui.pushButton.clicked.connect(self.noise_gen)
        self.ui.pushButton_2.clicked.connect(self.start_model)

    def file_browse(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, "Select image", filter="Image files (*.jpg *.png)")
        self.ui.lineEdit_filename.setText(fileName[0])
        print(fileName[0])

    def noise_gen(self):
        start(self.ui.lineEdit_filename.text())

    def start_model(self):
        if(self.ui.checkBox.isChecked()):
            start_from_gui(self.ui.lineEdit_filename.text())

if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    window = mainWindow()
    window.show()

    app.exec_()
