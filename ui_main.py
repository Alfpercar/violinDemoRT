# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_main.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(993, 692)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.pbLevel = QtGui.QProgressBar(self.centralwidget)
        self.pbLevel.setMaximum(1000)
        self.pbLevel.setProperty("value", 123)
        self.pbLevel.setTextVisible(False)
        self.pbLevel.setOrientation(QtCore.Qt.Vertical)
        self.pbLevel.setObjectName(_fromUtf8("pbLevel"))
        self.horizontalLayout.addWidget(self.pbLevel)
        # #pitch bar
        # self.pbPitch = QtGui.QProgressBar(self.centralwidget)
        # self.pbPitch.setMaximum(1500)
        # self.pbPitch.setProperty("value", 0)
        # self.pbPitch.setTextVisible(False)
        # self.pbPitch.setOrientation(QtCore.Qt.Vertical)
        # self.pbPitch.setObjectName(_fromUtf8("pbPitch"))
        # self.horizontalLayout.addWidget(self.pbPitch)

        # # vertical pane to put pitch and string labels
        # self.pane_pitch_string = QtGui.QFrame(self.centralwidget)
        # self.pane_pitch_string.setFrameShape(QtGui.QFrame.NoFrame)
        # self.pane_pitch_string.setFrameShadow(QtGui.QFrame.Plain)
        # self.pane_pitch_string.setObjectName(_fromUtf8("pane_pitch_string"))
        # self.verticalLayout1 = QtGui.QVBoxLayout(self.pane_pitch_string)
        # self.label_pitch = QtGui.QLabel(self.pane_pitch_string) #label with computed pitch
        # self.label_pitch.setObjectName(_fromUtf8("label_pitch"))
        # self.verticalLayout1.addWidget(self.label_pitch)
        # self.label_string = QtGui.QLabel(self.pane_pitch_string)  # label with estimated string
        # self.label_string.setObjectName(_fromUtf8("label_pitch"))
        # self.verticalLayout1.addWidget(self.label_string)
        # self.horizontalLayout.addWidget(self.pane_pitch_string)

        #two frames with verticalLayout
        self.frame1 = QtGui.QFrame(self.centralwidget)
        self.frame1.setFrameShape(QtGui.QFrame.NoFrame)
        self.frame1.setFrameShadow(QtGui.QFrame.Plain)
        self.frame1.setObjectName(_fromUtf8("frame1"))
        self.verticalLayout1 = QtGui.QVBoxLayout(self.frame1)
        # self.verticalLayout1.setContentsMargin(0)
        self.verticalLayout1.setObjectName(_fromUtf8("verticalLayout1"))
        self.frame2 = QtGui.QFrame(self.centralwidget)
        self.frame2.setFrameShape(QtGui.QFrame.NoFrame)
        self.frame2.setFrameShadow(QtGui.QFrame.Plain)
        self.frame2.setObjectName(_fromUtf8("frame2"))
        self.verticalLayout2 = QtGui.QVBoxLayout(self.frame2)
        # self.verticalLayout2.setContentsMargin(0)
        self.verticalLayout2.setObjectName(_fromUtf8("verticalLayout2"))


        # --add a first plot graph for FFT
        self.label_harmEnv = QtGui.QLabel(self.frame1)
        self.label_harmEnv.setObjectName(_fromUtf8("label_harmEnv"))
        self.verticalLayout1.addWidget(self.label_harmEnv)
        self.grFFT = PlotWidget(self.frame1)
        self.grFFT.setObjectName(_fromUtf8("grFFT"))
        self.verticalLayout1.addWidget(self.grFFT)
        # --add a plot graph for PCM
        self.label_PCM = QtGui.QLabel(self.frame1)
        self.label_PCM.setObjectName(_fromUtf8("label_PCM"))
        self.verticalLayout1.addWidget(self.label_PCM)
        self.grPCM = PlotWidget(self.frame1)
        self.grPCM.setObjectName(_fromUtf8("grPCM"))
        self.verticalLayout1.addWidget(self.grPCM)
        # --add a plot graph for pitch
        self.label_pitch = QtGui.QLabel(self.frame1)
        self.label_pitch.setObjectName(_fromUtf8("label_pitch"))
        self.verticalLayout1.addWidget(self.label_pitch)
        self.grPitch = PlotWidget(self.frame1)
        self.grPitch.setObjectName(_fromUtf8("grPitch"))
        self.verticalLayout1.addWidget(self.grPitch)




        #--add a plot graph for Velocity
        self.label_velocity = QtGui.QLabel(self.frame2)
        self.label_velocity.setObjectName(_fromUtf8("label_velocity"))
        self.verticalLayout2.addWidget(self.label_velocity)
        self.grVelocity = PlotWidget(self.frame1)
        self.grVelocity.setObjectName(_fromUtf8("grVelocity"))
        self.verticalLayout2.addWidget(self.grVelocity)
        #--add a 4th plot graph for Force
        self.label_force = QtGui.QLabel(self.frame2)
        self.label_force.setObjectName(_fromUtf8("label_force"))
        self.verticalLayout2.addWidget(self.label_force)
        self.grForce = PlotWidget(self.frame2)
        self.grForce.setObjectName(_fromUtf8("grForce"))
        self.verticalLayout2.addWidget(self.grForce)
        #--add a 5th plot graph for BBD - change for String prediction probabilities (one-hot)
        self.label_bbd = QtGui.QLabel(self.frame2)
        self.label_bbd.setObjectName(_fromUtf8("label_bbd"))
        self.verticalLayout2.addWidget(self.label_bbd)
        self.grBBD = PlotWidget(self.frame2)
        self.grBBD.setObjectName(_fromUtf8("grBBD"))
        self.verticalLayout2.addWidget(self.grBBD)
        # --add a 6th plot graph for played string
        self.label_string = QtGui.QLabel(self.frame2)
        self.label_string.setObjectName(_fromUtf8("label_string"))
        self.verticalLayout2.addWidget(self.label_string)
        self.grString = PlotWidget(self.frame1)
        self.grString.setObjectName(_fromUtf8("grString"))
        self.verticalLayout2.addWidget(self.grString)


        #--
        self.horizontalLayout.addWidget(self.frame1)
        self.horizontalLayout.addWidget(self.frame2)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label_harmEnv.setText(_translate("MainWindow", "Harmonic Envelope:", None))
        self.label_PCM.setText(_translate("MainWindow", "raw data (PCM):", None))
        self.label_velocity.setText(_translate("MainWindow", "Bowing Velocity:", None))
        self.label_force.setText(_translate("MainWindow", "Bowing Force:", None))
        self.label_bbd.setText(_translate("MainWindow", "one_hot String:", None)) #"Bow-bridge distance:", None))
        self.label_string.setText(_translate("MainWindow", "String:", None))
        self.label_pitch.setText(_translate("MainWindow", "Pitch:", None))

        #self.label_pitch.setText(_translate("MainWindow", "1000", None))
        #self.label_pitch.setFont(QtGui.QFont("Times", 18, QtGui.QFont.Bold))
        #self.label_string.setText(_translate("MainWindow", "G", None))
        #self.label_string.setFont(QtGui.QFont("Times", 25, QtGui.QFont.Bold))

from pyqtgraph import PlotWidget
