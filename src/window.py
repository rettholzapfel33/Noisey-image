# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/qt_designer_file/main_window_new.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1554, 932)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.original = Label(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.original.sizePolicy().hasHeightForWidth())
        self.original.setSizePolicy(sizePolicy)
        self.original.setText("")
        self.original.setScaledContents(True)
        self.original.setObjectName("original")
        self.verticalLayout_2.addWidget(self.original)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_2.addWidget(self.label_8)
        self.verticalLayout_2.setStretch(0, 5)
        self.verticalLayout_5.addLayout(self.verticalLayout_2)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.original_2 = Label(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.original_2.sizePolicy().hasHeightForWidth())
        self.original_2.setSizePolicy(sizePolicy)
        self.original_2.setText("")
        self.original_2.setScaledContents(True)
        self.original_2.setObjectName("original_2")
        self.verticalLayout_4.addWidget(self.original_2)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.verticalLayout_4.addWidget(self.label_10)
        self.verticalLayout_4.setStretch(0, 5)
        self.verticalLayout_5.addLayout(self.verticalLayout_4)
        self.gridLayout.addLayout(self.verticalLayout_5, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 2, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setStyleSheet("background-color: rgb(114, 159, 207);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_5.addWidget(self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setStyleSheet("background-color: rgb(0, 223, 255);")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_5.addWidget(self.pushButton)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setStyleSheet("background-color: rgb(239, 41, 41);")
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_5.addWidget(self.pushButton_4)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.horizontalLayout_5.setStretch(0, 5)
        self.horizontalLayout_5.setStretch(1, 4)
        self.horizontalLayout_5.setStretch(2, 4)
        self.horizontalLayout_5.setStretch(3, 2)
        self.horizontalLayout_5.setStretch(4, 2)
        self.horizontalLayout_5.setStretch(5, 1)
        self.gridLayout_2.addLayout(self.horizontalLayout_5, 3, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout.addWidget(self.comboBox)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setObjectName("checkBox_2")
        self.horizontalLayout_2.addWidget(self.checkBox_2)
        self.compoundAug = QtWidgets.QCheckBox(self.centralwidget)
        self.compoundAug.setObjectName("compoundAug")
        self.horizontalLayout_2.addWidget(self.compoundAug)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setEnabled(True)
        self.progressBar.setMaximum(5)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_2.addWidget(self.progressBar)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 3, 2, 1, 1)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 1, 1, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout.addWidget(self.line_2, 1, 2, 1, 1)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.preview = Label(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.preview.sizePolicy().hasHeightForWidth())
        self.preview.setSizePolicy(sizePolicy)
        self.preview.setText("")
        self.preview.setScaledContents(True)
        self.preview.setObjectName("preview")
        self.verticalLayout_3.addWidget(self.preview)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_3.addWidget(self.label_9)
        self.verticalLayout_3.setStretch(0, 5)
        self.verticalLayout_7.addLayout(self.verticalLayout_3)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.preview_2 = Label(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.preview_2.sizePolicy().hasHeightForWidth())
        self.preview_2.setSizePolicy(sizePolicy)
        self.preview_2.setText("")
        self.preview_2.setScaledContents(True)
        self.preview_2.setObjectName("preview_2")
        self.verticalLayout_6.addWidget(self.preview_2)
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_6.addWidget(self.label_11)
        self.verticalLayout_6.setStretch(0, 5)
        self.verticalLayout_7.addLayout(self.verticalLayout_6)
        self.gridLayout.addLayout(self.verticalLayout_7, 0, 2, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.tab)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.listWidget = QtWidgets.QListWidget(self.tab)
        self.listWidget.setObjectName("listWidget")
        self.horizontalLayout_7.addWidget(self.listWidget)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.tab_2)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.fileList = QtWidgets.QListWidget(self.tab_2)
        self.fileList.setObjectName("fileList")
        self.horizontalLayout_8.addWidget(self.fileList)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 2, 1)
        self.progressBar_2 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_2.setProperty("value", 0)
        self.progressBar_2.setObjectName("progressBar_2")
        self.gridLayout.addWidget(self.progressBar_2, 2, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.listAugs = QtWidgets.QListWidget(self.centralwidget)
        self.listAugs.setObjectName("listAugs")
        self.verticalLayout_13.addWidget(self.listAugs)
        self.horizontalLayout_4.addLayout(self.verticalLayout_13)
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.upListAug = QtWidgets.QPushButton(self.centralwidget)
        self.upListAug.setObjectName("upListAug")
        self.verticalLayout_11.addWidget(self.upListAug)
        self.downListAug = QtWidgets.QPushButton(self.centralwidget)
        self.downListAug.setObjectName("downListAug")
        self.verticalLayout_11.addWidget(self.downListAug)
        self.deleteListAug = QtWidgets.QPushButton(self.centralwidget)
        self.deleteListAug.setObjectName("deleteListAug")
        self.verticalLayout_11.addWidget(self.deleteListAug)
        self.horizontalLayout_4.addLayout(self.verticalLayout_11)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.addAug = QtWidgets.QPushButton(self.centralwidget)
        self.addAug.setObjectName("addAug")
        self.verticalLayout_10.addWidget(self.addAug)
        self.loadAug = QtWidgets.QPushButton(self.centralwidget)
        self.loadAug.setObjectName("loadAug")
        self.verticalLayout_10.addWidget(self.loadAug)
        self.saveAug = QtWidgets.QPushButton(self.centralwidget)
        self.saveAug.setObjectName("saveAug")
        self.verticalLayout_10.addWidget(self.saveAug)
        self.horizontalLayout_3.addLayout(self.verticalLayout_10)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.horizontalLayout_3.addLayout(self.verticalLayout_9)
        self.demoAug = QtWidgets.QPushButton(self.centralwidget)
        self.demoAug.setObjectName("demoAug")
        self.horizontalLayout_3.addWidget(self.demoAug)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_3)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.gridLayout.addLayout(self.verticalLayout, 3, 1, 1, 1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.imageButton = QtWidgets.QRadioButton(self.centralwidget)
        self.imageButton.setObjectName("imageButton")
        self.gridLayout_3.addWidget(self.imageButton, 0, 0, 1, 1)
        self.videoButton = QtWidgets.QRadioButton(self.centralwidget)
        self.videoButton.setObjectName("videoButton")
        self.gridLayout_3.addWidget(self.videoButton, 1, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_3, 3, 0, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 6)
        self.gridLayout.setColumnStretch(2, 6)
        self.horizontalLayout_6.addLayout(self.gridLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1554, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuFont_Size = QtWidgets.QMenu(self.menuEdit)
        self.menuFont_Size.setObjectName("menuFont_Size")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_As = QtWidgets.QAction(MainWindow)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionIncrease_Size = QtWidgets.QAction(MainWindow)
        self.actionIncrease_Size.setObjectName("actionIncrease_Size")
        self.actionDecrease_Size = QtWidgets.QAction(MainWindow)
        self.actionDecrease_Size.setObjectName("actionDecrease_Size")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFont_Size.addAction(self.actionIncrease_Size)
        self.menuFont_Size.addAction(self.actionDecrease_Size)
        self.menuEdit.addAction(self.menuFont_Size.menuAction())
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_8.setText(_translate("MainWindow", "Original Image"))
        self.label_10.setText(_translate("MainWindow", "Segmentation"))
        self.label_2.setText(_translate("MainWindow", "Augmentation Generator"))
        self.label.setText(_translate("MainWindow", "Models"))
        self.pushButton_2.setText(_translate("MainWindow", "Run model"))
        self.pushButton.setText(_translate("MainWindow", "Run preview"))
        self.pushButton_4.setText(_translate("MainWindow", "Exit"))
        self.label_3.setText(_translate("MainWindow", "Select a model:"))
        self.checkBox_2.setText(_translate("MainWindow", "Display result on a separate window"))
        self.compoundAug.setText(_translate("MainWindow", "Compound Augmentations"))
        self.label_9.setText(_translate("MainWindow", "Noisy"))
        self.label_11.setText(_translate("MainWindow", "Overlay"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Items"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Images"))
        self.upListAug.setText(_translate("MainWindow", "^"))
        self.downListAug.setText(_translate("MainWindow", "v"))
        self.deleteListAug.setText(_translate("MainWindow", "x"))
        self.addAug.setText(_translate("MainWindow", "Add Augmentations"))
        self.loadAug.setText(_translate("MainWindow", "Load Augmentations"))
        self.saveAug.setText(_translate("MainWindow", "Save Augmentations"))
        self.demoAug.setText(_translate("MainWindow", "Demo Augmentation"))
        self.imageButton.setText(_translate("MainWindow", "Images"))
        self.videoButton.setText(_translate("MainWindow", "Videos"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuFont_Size.setTitle(_translate("MainWindow", "Font Size"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As"))
        self.actionIncrease_Size.setText(_translate("MainWindow", "Increase Size"))
        self.actionDecrease_Size.setText(_translate("MainWindow", "Decrease Size"))

from src.qlabel import Label
