# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DisplayUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(908, 501)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.radioButtonCam = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButtonCam.setGeometry(QtCore.QRect(40, 380, 115, 19))
        self.radioButtonCam.setObjectName("radioButtonCam")
        self.radioButtonFile = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButtonFile.setGeometry(QtCore.QRect(40, 430, 115, 19))
        self.radioButtonFile.setObjectName("radioButtonFile")
        self.Open = QtWidgets.QPushButton(self.centralwidget)
        self.Open.setGeometry(QtCore.QRect(250, 400, 93, 28))
        self.Open.setObjectName("Open")
        self.Close = QtWidgets.QPushButton(self.centralwidget)
        self.Close.setGeometry(QtCore.QRect(410, 400, 93, 28))
        self.Close.setObjectName("Close")
        self.DisplayLabel = QtWidgets.QLabel(self.centralwidget)
        self.DisplayLabel.setGeometry(QtCore.QRect(30, 30, 576, 324))
        self.DisplayLabel.setText("")
        self.DisplayLabel.setObjectName("DisplayLabel")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(650, 0, 241, 471))
        self.textBrowser.setObjectName("textBrowser")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 908, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.radioButtonCam.setText(_translate("MainWindow", "摄像头"))
        self.radioButtonFile.setText(_translate("MainWindow", "选择文件"))
        self.Open.setText(_translate("MainWindow", "Open"))
        self.Close.setText(_translate("MainWindow", "Close"))
