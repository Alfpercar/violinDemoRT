from PyQt5 import QtGui, QtCore

import sys, os
#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../violinDemo/'))
import ui_main
import numpy as np
import pyqtgraph
import ListenViolin
import math
from matplotlib.mlab import find
import scipy.interpolate as inter


class MainWindow(QtGui.QMainWindow, ui_main.Ui_MainWindow):
    def __init__(self, parent=None):
        pyqtgraph.setConfigOption('background', 'w') #before loading widget
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.grFFT.plotItem.showGrid(True, True, 0.7)
        self.grPCM.plotItem.showGrid(True, True, 0.7)
        self.grVelocity.plotItem.showGrid(True, True, 0.7)
        self.grForce.plotItem.showGrid(True, True, 0.7)
        self.grBBD.plotItem.showGrid(True, True, 0.7)
        self.maxFFT=0
        self.maxPCM=1
        self.grPCM.plotItem.setRange(yRange=[-self.maxPCM,self.maxPCM])
        self.ear = ListenViolin.ListenViolin()
        self.ear.stream_start()

    def update(self):
        if not self.ear.audioToPlot is None: # and not self.ear.fft is None:
            pcmMax=np.max(np.abs(self.ear.audioToPlot))
            #print("pcmMax:", pcmMax)
            #if pcmMax>self.maxPCM:
            #    self.maxPCM=pcmMax
            #self.grPCM.plotItem.setRange(yRange=[-pcmMax,pcmMax])
       #     if np.max(self.ear.fft)>self.maxFFT:
       #     self.maxFFT=np.max(self.ear.fft)
                #self.minFFT=np.min(self.ear.fft)
       #         self.grFFT.plotItem.setRange(yRange=[-120,0])
            self.pbLevel.setValue(1000*pcmMax/self.maxPCM)
            #self.pbPitch.setValue(self.ear.pitch)
            #self.label_pitch.setText("%04d" % self.ear.pitch)
            # Plot waveform
            pen=pyqtgraph.mkPen(color='b')
            self.grPCM.plot(self.ear.datax, self.ear.audioToPlot,
                            pen=pen, clear=True)
            #plot FFT
            #pen=pyqtgraph.mkPen(color='g')
            #self.grFFT.plot(self.ear.fftx,self.ear.fft, pen=pen, clear=True)

            #plot harmonics
            # pen = pyqtgraph.mkPen(color='b')
            # idx = find(self.ear.hfreq > 0)
            # #if(len(idx)>0):
            # self.grFFT.plot(self.ear.hfreq[idx], self.ear.hmag[idx], pen=pen, symbol='x', clear=True)
            # self.grFFT.plotItem.setRange(yRange=[-120, 0], xRange=[0, 22050])

            #plot harmonic envelope
            #pen=pyqtgraph.mkPen(color='r')
            #self.grFFT.plot(self.ear.fftx,self.ear.harmonicEnvelope, pen=pen)
            #
            #plot energy Bands
            pen=pyqtgraph.mkPen('r', width=3, style=QtCore.Qt.DashLine)
            if(len(self.ear.energyBand)>0):
                self.grFFT.plot(range(0, self.ear.energyBand.shape[0]), self.ear.energyBand, pen=pen, clear=True) #self.ear.bandCentersHz,
                self.grFFT.plotItem.setRange(yRange=[0, 2]) # , xRange=[0, 22050])
            else:
                self.grFFT.plot(range(0, self.ear.MelSpec.shape[0]), self.ear.MelSpec[:, 1], pen=pen, clear=True)

            # plot pitch
            pen = pyqtgraph.mkPen('b')
            self.grPitch.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.f0BufferToPlot, clear=True, pen=pen)
            self.grPitch.plotItem.setRange(yRange=[0, 1500])


            # plot bowing velocity --> or --> string_2
            pen = pyqtgraph.mkPen('k', style=QtCore.Qt.DashLine)
            #self.grVelocity.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)),
            #                   np.ones(self.ear.numPredictedValuesToPlot), clear=True, pen=pen)
            #self.grVelocity.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)),
            #                   np.ones(self.ear.numPredictedValuesToPlot)*2, clear=False, pen=pen)
            #self.grVelocity.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)),
            #                   np.ones(self.ear.numPredictedValuesToPlot)*3, clear=False, pen=pen)
            #self.grVelocity.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)),
            #                   np.ones(self.ear.numPredictedValuesToPlot)*4, clear=False, pen=pen)
            pen = pyqtgraph.mkPen('g', width=1,)  # , width=3, style=QtCore.Qt.DashLine)
            self.grVelocity.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.predVelocityToPlot, clear=True, pen=pen)
            pen = pyqtgraph.mkPen('r', width=1)  # , width=3, style=QtCore.Qt.DashLine)
            self.grVelocity.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.rmsE_toPlot, clear=False, pen=pen)
            pen = pyqtgraph.mkPen('k', width=1)  # , width=3, style=QtCore.Qt.DashLine)
            self.grVelocity.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.rmsE_toPlot_2, clear=False, pen=pen)
            self.grVelocity.plotItem.setRange(yRange=[0, 1])

            # plot bowing force --> string_2 one-hot encoding
            pen = pyqtgraph.mkPen('b') #, width=3, style=QtCore.Qt.DashLine)
            pen = pyqtgraph.mkPen('k') #, width=3, style=QtCore.Qt.DashLine)
            self.grForce.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.predOneHotStringToPlot_2[0, :], clear = True, pen = pen)
            pen = pyqtgraph.mkPen('g', width=2)  # , width=3, style=QtCore.Qt.DashLine)
            self.grForce.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.predOneHotStringToPlot_2[1, :] , clear=False, pen=pen)
            pen = pyqtgraph.mkPen('b', width=2)  # , width=3, style=QtCore.Qt.DashLine)
            self.grForce.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.predOneHotStringToPlot_2[2, :], clear=False, pen=pen)
            pen = pyqtgraph.mkPen('r', width=2)  # , width=3, style=QtCore.Qt.DashLine)
            self.grForce.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.predOneHotStringToPlot_2[3, :], clear=False, pen=pen)
            pen = pyqtgraph.mkPen(color=(255, 129, 0), width=2)  # , width=3, style=QtCore.Qt.DashLine)
            self.grForce.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.predOneHotStringToPlot_2[4, :], clear=False, pen=pen)

            # plot one-hot string probabilities
            pen = pyqtgraph.mkPen('k') #, width=3, style=QtCore.Qt.DashLine)
            self.grBBD.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.predOneHotStringToPlot[0, :], clear = True, pen = pen)
            pen = pyqtgraph.mkPen('g', width=2)  # , width=3, style=QtCore.Qt.DashLine)
            self.grBBD.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.predOneHotStringToPlot[1, :] , clear=False, pen=pen)
            pen = pyqtgraph.mkPen('b', width=2)  # , width=3, style=QtCore.Qt.DashLine)
            self.grBBD.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.predOneHotStringToPlot[2, :], clear=False, pen=pen)
            pen = pyqtgraph.mkPen('r', width=2)  # , width=3, style=QtCore.Qt.DashLine)
            self.grBBD.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.predOneHotStringToPlot[3, :], clear=False, pen=pen)
            pen = pyqtgraph.mkPen(color=(255, 129, 0), width=2)  # , width=3, style=QtCore.Qt.DashLine)
            self.grBBD.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.predOneHotStringToPlot[4, :], clear=False, pen=pen)
            #self.grBBD.plotItem.setRange(yRange=[0, 1])


            # plot which string
            pen = pyqtgraph.mkPen('k', style=QtCore.Qt.DashLine)
            self.grString.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)),
                               np.ones(self.ear.numPredictedValuesToPlot), clear=True, pen=pen)
            self.grString.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)),
                               np.ones(self.ear.numPredictedValuesToPlot)*2, clear=False, pen=pen)
            self.grString.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)),
                               np.ones(self.ear.numPredictedValuesToPlot)*3, clear=False, pen=pen)
            self.grString.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)),
                               np.ones(self.ear.numPredictedValuesToPlot)*4, clear=False, pen=pen)
            pen = pyqtgraph.mkPen('g', width=2,)  # , width=3, style=QtCore.Qt.DashLine)
            self.grString.plot(np.array(range(0, self.ear.numPredictedValuesToPlot)), self.ear.predStringToPlot,
                               clear=False, pen=pen)
            self.grString.plotItem.setRange(yRange=[-0.5, 4.2])

        QtCore.QTimer.singleShot(1, self.update) # QUICKLY repeat

if __name__=="__main__":
    app = QtGui.QApplication(sys.argv)
    form = MainWindow()
    form.show()
    form.update() #start with something
    app.exec_()
    print("DONE")
