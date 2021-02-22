from mainwindow_ui import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5 import QtCore

class mainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton_browse_file.clicked.connect(self.file_browse)


    def file_browse(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, "Select image", filter="Image files (*.jpg *.png)")
        self.ui.lineEdit_filename.setText(fileName[0])
        print(fileName[0])

if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    window = mainWindow()
    window.show()

    app.exec_()