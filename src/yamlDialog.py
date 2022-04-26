# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/qt_designer_file/dialogAug.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(645, 430)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 380, 591, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.listWidget = QtWidgets.QListWidget(Dialog)
        self.listWidget.setGeometry(QtCore.QRect(20, 50, 256, 251))
        self.listWidget.setObjectName("listWidget")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 30, 111, 17))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(300, 32, 111, 20))
        self.label_2.setObjectName("label_2")
        self.noiseRange = QtWidgets.QLineEdit(Dialog)
        self.noiseRange.setGeometry(QtCore.QRect(300, 276, 161, 25))
        self.noiseRange.setObjectName("noiseRange")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(301, 256, 161, 17))
        self.label_3.setObjectName("label_3")
        self.previewImage = QtWidgets.QLabel(Dialog)
        self.previewImage.setGeometry(QtCore.QRect(300, 50, 321, 181))
        self.previewImage.setText("")
        self.previewImage.setObjectName("previewImage")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(480, 257, 91, 17))
        self.label_5.setObjectName("label_5")
        self.exampleLine = QtWidgets.QLineEdit(Dialog)
        self.exampleLine.setGeometry(QtCore.QRect(479, 276, 91, 25))
        self.exampleLine.setObjectName("exampleLine")
        self.minimum = QtWidgets.QLineEdit(Dialog)
        self.minimum.setGeometry(QtCore.QRect(300, 330, 71, 22))
        self.minimum.setInputMask("")
        self.minimum.setObjectName("minimum")
        self.maximum = QtWidgets.QLineEdit(Dialog)
        self.maximum.setGeometry(QtCore.QRect(390, 330, 71, 22))
        self.maximum.setObjectName("maximum")
        self.increment = QtWidgets.QLineEdit(Dialog)
        self.increment.setGeometry(QtCore.QRect(480, 330, 91, 22))
        self.increment.setObjectName("increment")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(300, 310, 71, 16))
        self.label_4.setObjectName("label_4")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(390, 310, 71, 16))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setGeometry(QtCore.QRect(480, 310, 91, 16))
        self.label_7.setObjectName("label_7")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Augmentations:</span></p></body></html>"))
        self.label_2.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-weight:600;\">Preview:</span></p></body></html>"))
        self.label_3.setText(_translate("Dialog", "Noise (or Noise Range)"))
        self.label_5.setText(_translate("Dialog", "Example:"))
        self.label_4.setText(_translate("Dialog", "Minimum"))
        self.label_6.setText(_translate("Dialog", "Maximum"))
        self.label_7.setText(_translate("Dialog", "Increment"))

