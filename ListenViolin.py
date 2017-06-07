"""
this is a stripped down version of the ListenViolin class.

"""
import os, sys
import math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../computeDescriptorsDir'))
import pyaudio
import time
#import pylab
import numpy as np
import threading
from scipy.signal import get_window
from matplotlib.mlab import find
#import math
from computeDescriptors import energyInBands, computeHarmonicEnvelope
import tensorflow as tf

import harmonicModel as HM
#import utils as UF
#import sineModel as SM
from tf_model import TfModel
from SckitModels import SKitLearnModel

# Librosa for audio
import librosa

class ListenViolin(object):

    """
    The ListenViolin class is made to provide access to continuously recorded
    (and mathematically processed) microphone data.
    """

    def __init__(self,device=None,rate=None):
        # constants
        self.TF_MODEL = 1
        self.SCIKIT_MODEL = 2
        self.BAGGING_MODEL = 3
        self.bandCentersHz = np.array(
            [103, 171, 245, 326, 413, 508, 611, 722, 843, 975, 1117, 1272, 1439, 1621, 1819, 2033, 2266, 2518, 2792,
             3089, 3412, 3761, 4141, 4553, 5000, 5485, 6011, 6582, 7202, 7874, 8604, 9396, 10255, 11187, 12198, 13296,
             14487, 15779, 17181, 18703])

        """fire up the ListenViolin class."""
        self.p = pyaudio.PyAudio()
        self.chunk = 2048   # number of data points to read at a time
        self.minFFTVal = -120
        self.minEnergyNormVal = 0
        self.device = device
        self.rate = rate
        self.energyBand = np.ones(len(self.bandCentersHz)) * self.minEnergyNormVal
        self.fftx = []
        self.fft = []
        self.hmag = np.ones(len(self.bandCentersHz)) * self.minFFTVal
        self.hfreq = np.zeros(len(self.bandCentersHz))
        self.harmonicEnvelope = np.zeros(2049)
        self.pcmMax = 10000
        self.pitch = 0
        self.analysis_win_l = 2048  # this is the frame-length
        self.pitch_stability_threshold = 0.5
        self.prevf0 = 0
        self.audioBuffer = []
        self.prev_hfreqp = []

        self.string_model_type = self.TF_MODEL  # BAGGING_MODEL  #TF_MODEL  #BAGGING_MODEL       #
        self.doEnergyBands = False
        # MODELS - velocity
        path = '../TFModels/models/'
        if self.doEnergyBands:
            # ---- tensorFlow Model for Bow Velocity -----
            #model_name = 'velocity_convnet_20170214_1754'
            #model_name = 'velocity_convnet_40Bands_20170602_1549' #<-- this should work with Yahama's recs
            model_name = 'velocity_convnet_40Bands_20170602_1832' #model after downsampling to match analHop=1024
            #model_name = 'velocity_convnet_40Bands_hop25620170602_2328' ##model after downsampling to match analHop=1024
            predictionOP = 'pred/activationOperationForOutput'  # 'pred/Sigmoid'
            self.velocityModel = TfModel(path, model_name, predictionOP, self.minEnergyNormVal)
        else:
            model_name = 'velocity_convnet_MFCCs_hop12820170602_2347'
            #model_name = 'velocity_convnet_MFCCs_hop12820170606_1709'
            predictionOP = 'pred/activationOperationForOutput'  # 'pred/Sigmoid'
            self.velocityModel = TfModel(path, model_name, predictionOP, self.minEnergyNormVal)

        # MODELS - string
        #doString_TF = 1

        if self.string_model_type == self.TF_MODEL:
            print('TensorFlow Model')
            if self.doEnergyBands:
                print('Compute EnergyBands')
                # ---- tensorFlow Model for String detection -----
                #model_name = 'string_convnet_20170417_1204'     # mic.gliss
                #model_name = 'string_convnet_20170417_1328'     # mic.vibrato scale
                #model_name = 'string_ffnet_20170221_1437' # pickup_model: 'string_convnet_20170216_1441'
                #  cuerdas al aire: 'string_convnet_20170221_1642'         # 'string_ffnet_20170221_1437'
                # model_name = 'string_convnet_yamahaDB20170419_2309' #yamaha: NO va bien con recs 2017
                #model_name = 'string_convnet_recs2017_20170420_1946'  # funciona bastante bien con trainingRecs
                #model_name = 'string_convnet_recs2017_plus_Yamaha_20170420_0902'  # yamaha + recs2017 :  funciona bastante bien con trainingRecs
                #model_name = 'string_convnet_scarlett_MyV_hop256_20170421_1604'       #Scarlett_MyViolin with hopSize=256. GREAT!!!!! 'string_convnet_scarlett_MyV_hop512_20170421_1632' #
                #model_name = 'string_convnet_scarlett_MyV_hop256_20170505_0945'   #Long recs hop=256
                model_name = 'string_convnet_scarlett_MyV_hop512_20170505_1039' #Long recs hop=512 512 Funciona muy bien con trainingdB!
                #model_name = 'string_convnet_scarlett_MyV_hop1024_20170505_1100' #Long recs hop=1024. Funciona MEJOR
                #kk model_name = 'string_convnet_scarlett_MyV_hop1024_convHop50_20170505_1149'
                #kk model_name = 'string_convnet_scarlett_MyV_hop512_convHop25_20170505_1205'
                #model_name = 'string_convnet_scarlett_MyV_hop1024_convHop10_20170505_1215'
                #model_name = 'string_convnet_all_hopmixed_convHop10_20170506_1638' #all recs. fuciona regular con traindB
                predictionOP = 'pred/activationOperationForOutput'  # 'pred/Sigmoid'
                one_hot_encoding = True
                do_hysteresis = False
                self.stringModel = TfModel(path, model_name, predictionOP, self.minEnergyNormVal,
                                           self.velocityModel.inputWinLen, one_hot_encoding=one_hot_encoding,
                                           do_hysteresis=do_hysteresis)
            else:# melspectrogram
                print('Compute melspectrogram')
                #model_name = 'string_convnet_MelSpec_fftS2048_ffthop256_20170508_1713'
                #model_name = 'string_convnet_MelSpec_fftS1024_ffthop128_20170508_1826'
                model_name = 'string_convnet_MelSpec_fftS1024_ffthop128_20170508_1859'
                #model_name = 'string_convnet_MelSpec_fftS512_ffthop128_20170509_1814'
                predictionOP = 'pred/activationOperationForOutput'  # 'pred/Sigmoid'
                #minStringValue = 0
                one_hot_encoding = True
                do_hysteresis = True
                self.stringModel = TfModel(path, model_name, predictionOP, self.minEnergyNormVal, self.velocityModel.inputWinLen, one_hot_encoding=one_hot_encoding, do_hysteresis=do_hysteresis)
        elif self.string_model_type == self.SCIKIT_MODEL:
            print('Random Forests')
            model_name = 'string_randomForests_20170418_1826'  #'string_decissionTree_20170418_1710' #'string_decissionTree_20170418_1626'
            self.stringModel = SKitLearnModel(path, model_name)
        elif self.string_model_type == self.BAGGING_MODEL:
            print('Aggregation')
            # first model
            model_name = 'string_convnet_MelSpec_fftS1024_ffthop128_20170508_1859'
            predictionOP = 'pred/activationOperationForOutput'
            one_hot_encoding = True
            do_hysteresis = False
            self.stringModel_2 = TfModel(path, model_name, predictionOP, self.minEnergyNormVal,
                                       self.velocityModel.inputWinLen, one_hot_encoding=one_hot_encoding,
                                       do_hysteresis=do_hysteresis)
            self.bagging_prediction = np.zeros(0)
            self.bagging_prediction_one_hot = np.zeros(shape=(0, 5))

            #second model
            model_name = 'string_convnet_scarlett_MyV_hop512_20170505_1039'  # Long recs hop=512 512 Funciona muy bien con trainingdB!
            self.stringModel = TfModel(path, model_name, predictionOP, self.minEnergyNormVal,
                                       self.velocityModel.inputWinLen, one_hot_encoding=one_hot_encoding,
                                       do_hysteresis=do_hysteresis)
            self.bagging_prediction2 = np.zeros(0)
            self.bagging_prediction_one_hot2 = np.zeros(shape=(0, 5))

        self.numPredictedValuesToPlot = 900
        self.predVelocityToPlot = [0] * self.numPredictedValuesToPlot
        self.rmsE_toPlot = [0] * self.numPredictedValuesToPlot
        self.rmsE_toPlot_2 = [0] * self.numPredictedValuesToPlot
        self.predStringToPlot = [0]*self.numPredictedValuesToPlot
        self.predStringToPlot_2 = [0] * self.numPredictedValuesToPlot
        w, h = self.numPredictedValuesToPlot, 5
        self.predOneHotStringToPlot = [[0 for x in range(w)] for y in range(h)]
        self.predOneHotStringToPlot = np.array(self.predOneHotStringToPlot)
        self.predOneHotStringToPlot_2 = [[0 for x in range(w)] for y in range(h)]
        self.predOneHotStringToPlot_2 = np.array(self.predOneHotStringToPlot_2)
        self.f0BufferToPlot = [0] * self.numPredictedValuesToPlot
        self.f0buffer = [0] * self.numPredictedValuesToPlot
        self.f0bufferStability = [0] * 5

    ### SYSTEM TESTS

    def valid_low_rate(self,device):
        """set the rate to the lowest supported audio rate."""
        for testrate in [44100]:
            if self.valid_test(device,testrate):
                return testrate
        print("SOMETHING'S WRONG! I can't figure out how to use DEV", device)
        return None

    def valid_test(self,device,rate=44100):
        """given a device ID and a rate, return TRUE/False if it's valid."""
        try:
            self.info = self.p.get_device_info_by_index(device)
            if not self.info["maxInputChannels"] > 0:
                return False
            stream = self.p.open(format=pyaudio.paInt16, channels=1,
               input_device_index = device, frames_per_buffer=self.chunk,
               rate = int(self.info["defaultSampleRate"]),input=True)
            stream.close()
            return True
        except:
            return False

    def valid_input_devices(self):
        """
        See which devices can be opened for microphone input.
        call this when no PyAudio object is loaded.
        """
        mics = []
        for device in range(self.p.get_device_count()):
            if self.valid_test(device):
                mics.append(device)
        if len(mics)==0:
            print("no microphone devices found!")
        else:
            print("found %d microphone devices: %s" % (len(mics), mics))
        return mics

    ### SETUP AND SHUTDOWN

    def initiate(self):
        """run this after changing settings (like rate) before recording"""
        if self.device is None:
            self.list_devices()
            device = input('select device number: ')
            self.device = self.valid_input_devices()[int(device)] #pick one
        if self.rate is None:
            self.rate=self.valid_low_rate(self.device)
        if not self.valid_test(self.device, self.rate):
            print("guessing a valid microphone device/rate...")
            self.device=self.valid_input_devices()[0] #pick the first one
            self.rate=self.valid_low_rate(self.device)
        self.datax=np.arange(self.chunk)/float(self.rate)
        self.freqs = np.array(range(0, self.analysis_win_l)) / self.analysis_win_l * self.rate / 2
        msg='recording from "%s" '%self.info["name"]
        msg+='(device %d) '%self.device
        msg+='at %d Hz'%self.rate
        print(msg)

    def close(self):
        """gently detach from things."""
        print(" -- sending stream termination command...")
        self.keepRecording=False #the threads should self-close
        while(self.t.isAlive()): #wait for all threads to close
            time.sleep(.1)
        self.stream.stop_stream()
        self.p.terminate()

    ### ANALYSIS
    def stream_analysischunk(self):
        """reads some audio and re-launches itself"""
        try:
            chunk = np.fromstring(self.stream.read(self.chunk, exception_on_overflow=False), dtype=np.int16)/self.pcmMax
            self.audioToPlot = chunk
            self.audioBuffer = np.hstack((self.audioBuffer, chunk))

            # compute RMS energy as a descriptor correlated to bowing speed.
            rmsE_hop = 128 #1024
            rmsE_WinL =  256
            chunkL = len(chunk)
            # fill x from self.dataBuffer
            n_frames = int((chunkL - rmsE_WinL) / rmsE_hop) + 1
            #n_frames = chunkL // rmsE_hop
            startIdx=0
            #rmsE_chunk = np.zeros(n_frames)
            rmsE_toPlot_2 = self.rmsE_toPlot_2[:]
            for i_frame in range(0, n_frames):
                grain = chunk[startIdx:startIdx+rmsE_WinL]
                media = np.mean(grain)
                normE = grain - media
                frameE = np.sqrt(sum(pow(normE,2)/rmsE_WinL))
                #rmsE_chunk[i_frame] = frameE
                rmsE_toPlot_2.append(frameE)
                startIdx = startIdx + rmsE_hop

            rmsE_toPlot_2 = rmsE_toPlot_2[n_frames:]
            self.rmsE_toPlot_2 = rmsE_toPlot_2[:]



            if self.string_model_type == self.BAGGING_MODEL:
                [predStringToPlot, predOneHotStringToPlot, prediction, prediction_OneHot] = self.energyBandAnalysis(self.stringModel,
                                                                                     self.predStringToPlot,
                                                                                     self.predOneHotStringToPlot)
                if len(prediction):
                    self.bagging_prediction = np.hstack((self.bagging_prediction, prediction))
                    self.bagging_prediction_one_hot = np.vstack((self.bagging_prediction_one_hot, prediction_OneHot))
                #self.predStringToPlot = predStringToPlot
                #self.predOneHotStringToPlot = predOneHotStringToPlot
                [predStringToPlot, predOneHotStringToPlot, prediction2, prediction_OneHot2] = self.melSpectrogramAnalisys(self.stringModel_2,
                                                                                         self.predStringToPlot,
                                                                                         self.predOneHotStringToPlot)
                if len(prediction2):
                    self.bagging_prediction2 = np.hstack((self.bagging_prediction2, prediction2))
                    self.bagging_prediction_one_hot2 = np.vstack((self.bagging_prediction_one_hot2, prediction_OneHot2))
                #self.predStringToPlot_2 = predStringToPlot
                #self.predOneHotStringToPlot_2 = predOneHotStringToPlot
                len1 = len(self.bagging_prediction)
                len2 = len(self.bagging_prediction2)
                if( len1 > 0 and len2 > 0):
                    combined_result = 0
                    combined_one_hot_prob = np.ones(shape=(2, 5))
                    combined_one_hot_prob[0, :] = np.mean(self.bagging_prediction_one_hot, 0)
                    combined_one_hot_prob[1, :] = np.mean(self.bagging_prediction_one_hot2, 0)
                    max_prob = np.max(combined_one_hot_prob, axis=0)
                    combined_result = np.argmax(combined_one_hot_prob)
                    predStringToPlot = self.predStringToPlot
                    predStringToPlot = np.hstack((predStringToPlot,  combined_result))
                    predStringToPlot = predStringToPlot[1:]
                    self.predStringToPlot = predStringToPlot
                    predOneHotStringToPlot = self.predOneHotStringToPlot
                    predOneHotStringToPlot = np.vstack((predOneHotStringToPlot.T,  max_prob))
                    predOneHotStringToPlot = predOneHotStringToPlot[1:,:]
                    self.predOneHotStringToPlot = predOneHotStringToPlot.T


                    self.bagging_prediction = self.bagging_prediction[len1:]
                    self.bagging_prediction_one_hot = self.bagging_prediction_one_hot[len1:, :]
                    self.bagging_prediction2 = self.bagging_prediction2[len2:]
                    self.bagging_prediction_one_hot2 = self.bagging_prediction_one_hot2[len2:, :]


            else:
                if self.doEnergyBands:
                    [predStringToPlot, predOneHotStringToPlot, filteredPrediction, prediction_OneHot] = self.energyBandAnalysis(self.stringModel, self.predStringToPlot, self.predOneHotStringToPlot)
                    self.predStringToPlot = predStringToPlot
                    self.predOneHotStringToPlot = predOneHotStringToPlot
                else:
                    #print('melSpectrogramAnalisys()')
                    [predStringToPlot, predOneHotStringToPlot, filteredPrediction, prediction_OneHot] = self.melSpectrogramAnalisys(self.stringModel, self.predStringToPlot, self.predOneHotStringToPlot)
                    self.predStringToPlot_2 = predStringToPlot
                    self.predOneHotStringToPlot_2 = predOneHotStringToPlot

                # except exception_on_overflow:
            #    pass

        except Exception as E:
            print(" -- exception! terminating...")
            print(E, "\n" * 5)
            self.keepRecording = False
        if self.keepRecording:
            self.stream_thread_new()
        else:
            self.stream.close()
            self.p.terminate()
            print(" -- stream STOPPED")


    def melSpectrogramAnalisys(self, stringModel, predStringToPlot, predOneHotStringToPlot):
        #hopSize = 512  # 1024:esta funciona  # 183.75
        #windowType = 'blackman'
        #window = get_window(windowType, self.analysis_win_l)
        fftSize = 1024 #2048
        fft_hopSize = 128 #256
        #windowSize = fftSize #len(self.audioToPlot)
        #analHopSize = windowSize

        # fill x from self.dataBuffer
        num_frames = int((len(self.audioBuffer) - fftSize) / fft_hopSize) + 1
        x = self.audioBuffer[0:fftSize + (num_frames - 1) * fft_hopSize]
        samples_to_push = num_frames * fft_hopSize
        self.audioBuffer = self.audioBuffer[samples_to_push:]

        # Mel-scaled power (energy-squared) spectrogram
        Sxx = librosa.feature.melspectrogram(x, sr=self.rate, n_mels=128, n_fft=fftSize, hop_length=fft_hopSize)
        # n_fft=2048, hop_length=512,power=2.0,
        # Convert to log scale (dB). We'll use the peak power as reference.
        Sxx=Sxx[:, 0:num_frames]
        SxxdB = librosa.logamplitude(Sxx, ref_power=np.max)
        auxavg = np.average(SxxdB, axis=0)
        SxxdBNorm = SxxdB / auxavg
        SxxdBNorm_vel = SxxdB

        num_new_frames = SxxdBNorm.shape[1]
        self.MelSpec = SxxdBNorm
        self.energyBand = SxxdBNorm[:, 1]

        #filteredPrediction = []
        #prediction_OneHot = []

        for iFrame in range(0, num_new_frames):
            stringModel.addFrame(self.MelSpec[:, iFrame], keepLength=False)
            self.velocityModel.addFrame(SxxdBNorm_vel[:, iFrame], keepLength=False)

        [filteredPrediction_vel, _] = self.velocityModel.processNextBatch(num_new_frames)
        [filteredPrediction, prediction_OneHot] = stringModel.processNextBatch(num_new_frames)

        diff = self.velocityModel.maxPredValuesToStore - self.numPredictedValuesToPlot

        if diff > 0:
            self.predVelocityToPlot = self.velocityModel.outputBuffer[diff:]
            predStringToPlot = stringModel.outputBuffer[diff:]
            predOneHotStringToPlot = stringModel.outputBufferOneHot[:, diff:]

        else:
            self.predVelocityToPlot = self.velocityModel.outputBuffer
            predStringToPlot = stringModel.outputBuffer
            predOneHotStringToPlot = stringModel.one_hot_prediction

        return predStringToPlot, predOneHotStringToPlot, filteredPrediction, prediction_OneHot

    def energyBandAnalysis(self, stringModel, predStringToPlot, predOneHotStringToPlot):
            #fs = self.rate
            NyqFreq = self.rate/2
            fftSize = 2048
            hopSize = 1024 #1024:esta funciona  # 183.75
            windowType = 'blackman'
            window = get_window(windowType, self.analysis_win_l)
            # detect harmonics of input sound
            minSineDur = 0.1
            nHarmonics = 100
            minf0 = 180
            maxf0 = 1500
            f0et = 2               # f0et: error threshold in the f0 detection (ex: 5)
            harmDevSlope = 0.01

            # -----------
            # fill x from self.dataBuffer
            num_frames = int((len(self.audioBuffer) - window.size) / hopSize) + 1
            x = self.audioBuffer[0:window.size + (num_frames - 1) * hopSize]
            samples_to_push = num_frames * hopSize
            self.audioBuffer = self.audioBuffer[samples_to_push:]

            # # ------------
            window = window / sum(window)  # normalize analysis window
            hfreqp = self.prev_hfreqp  # initialize harmonic frequencies of previous frame
            # f0t = 0  # initialize f0 track ?????????
            f0stable = self.prevf0  # initialize f0 stable
            #print("f0stable:", f0stable)
            #while pin <= pend:
            pin = 0
            for iFrame in range(0, num_frames):
                #x1 = x[pin - hM1:pin + hM2]
                x1 = x[pin: pin + window.size]
                useTWM = 0
                mX, f0stable, f0t, hfreq, hmag, hphase = HM.harmonicModelAnalFrame(x1, window, fftSize, self.minFFTVal, self.rate,
                                                                                hfreqp, f0et, minf0, maxf0, nHarmonics,
                                                                                f0stable, harmDevSlope, useTWM)
                hfreqp = hfreq  # hfreq(previous)
                if pin == 0:  # first frame
                    xhfreq = np.array([hfreq])
                    xhmag = np.array([hmag])
                    xhphase = np.array([hphase])
                else:  # next frames
                    xhfreq = np.vstack((xhfreq, np.array([hfreq])))
                    xhmag = np.vstack((xhmag, np.array([hmag])))
                    xhphase = np.vstack((xhphase, np.array([hphase])))
                pin += hopSize  # advance sound pointer
           #xhfreq = SM.cleaningSineTracks(xhfreq,
           #                                round(self.rate * minSineDur / hopSize))  # delete tracks shorter than minSineDur
            # ------------
            self.prev_hfreqp = hfreq
            self.prevf0 = f0stable

            self.hfreq = xhfreq[0, :]
            self.hmag = xhmag[0, :]
            self.f0bufferStability = np.hstack((self.f0bufferStability, f0stable))
            self.f0bufferStability = self.f0bufferStability[1:]

            #print("current_f0:",current_f0)

            idx = find(self.f0bufferStability[-5:] > 0)
            if len(idx) > 0:
                pitch_stability = np.abs(f0stable-np.mean(self.f0bufferStability[idx]))/f0stable
            else:
                pitch_stability = 0
            #print("pitch_deviation", pitch_stability)

            if self.velocityModel.inputWinLen - xhfreq.shape[0] > 0:  # if number of input/output frames in the model > new frames (usual case)
                num_new_frames = xhfreq.shape[0]
            else:
                num_new_frames = self.velocityModel.inputWinLen  # use predictionWL frames and DONT USE the rest in hfreq

            #idx = find(self.hfreq > 0)
            #if len(idx) > 3:
            #print("f0stable:", f0stable, "pitch_stability:", pitch_stability, "idx:", idx)
            filteredPrediction = []
            prediction_OneHot = []

            if True: #f0stable > minf0 and f0stable < maxf0 and len(idx) > 4 and pitch_stability < self.pitch_stability_threshold:
                self.pitch = np.mean(self.f0bufferStability[-5:])
                self.f0buffer = np.hstack((self.f0buffer, f0stable))
                self.f0buffer = self.f0buffer[1:]
                diff = len(self.f0buffer) - self.numPredictedValuesToPlot
                if diff > 0:
                    self.f0BufferToPlot = self.f0buffer[diff:]
                else:
                    self.f0BufferToPlot = self.f0buffer

                rmsE_toPlot = self.rmsE_toPlot[:]
                for iFrame in range(0, num_new_frames):
                    self.harmonicEnvelope = computeHarmonicEnvelope(xhfreq[iFrame, :], xhmag[iFrame, :], NyqFreq, self.minFFTVal, fftSize, self.freqs)
                    energyBand_dB = energyInBands(self.harmonicEnvelope, self.bandCentersHz, self.rate, self.minFFTVal)
                    energy_bands = 10 ** (energyBand_dB / 20)
                    rmsEnergy_dB = 20 * np.log10(np.sqrt(np.mean(energy_bands ** 2, 0)))
                    energy_bands_norm = energyBand_dB / rmsEnergy_dB
                    energy_bands_norm = energy_bands_norm / 4
                    energy_bands_norm_4velocity = (energyBand_dB /120 )+1
                    self.energyBand = energy_bands_norm_4velocity #energy_bands_norm
                    self.velocityModel.addFrame(energy_bands_norm_4velocity, keepLength=False)
                    stringModel.addFrame(energy_bands_norm, keepLength=False)
                    rmsE_toPlot.append((rmsEnergy_dB/120)+1)

                rmsE_toPlot = rmsE_toPlot[num_new_frames:]
                self.rmsE_toPlot = rmsE_toPlot[:]

                self.velocityModel.processNextBatch(num_new_frames)
                [filteredPrediction, prediction_OneHot ] = stringModel.processNextBatch(num_new_frames)

                diff = self.velocityModel.maxPredValuesToStore - self.numPredictedValuesToPlot
                if diff > 0:
                    self.predVelocityToPlot = self.velocityModel.outputBuffer[diff:]
                    predStringToPlot = stringModel.outputBuffer[diff:]
                    predOneHotStringToPlot = stringModel.outputBufferOneHot[:, diff:]


                else:
                    self.predVelocityToPlot = self.velocityModel.outputBuffer
                    predStringToPlot = stringModel.outputBuffer
                    predOneHotStringToPlot = stringModel.one_hot_prediction
            else:
                self.pitch = 0
                self.f0buffer = np.hstack((self.f0buffer, 0))
                self.f0buffer = self.f0buffer[1:]
                diff = len(self.f0buffer) - self.numPredictedValuesToPlot
                if diff > 0:
                    self.f0BufferToPlot = self.f0buffer[diff:]
                else:
                    self.f0BufferToPlot = self.f0buffer
                #fill with zeros
                for iFrame in range(0, num_new_frames):
                    self.energyBand = np.ones(len(self.bandCentersHz)) * self.minFFTVal
                    self.velocityModel.addFrame(self.energyBand)
                    stringModel.addFrame(self.energyBand)

                self.velocityModel.processNextBatch(num_new_frames)
                stringModel.processNextBatch(num_new_frames)
                diff = self.velocityModel.maxPredValuesToStore - self.numPredictedValuesToPlot
                if diff > 0:
                    self.predVelocityToPlot = self.velocityModel.outputBuffer[diff:]
                    predStringToPlot = stringModel.outputBuffer[diff:]
                else:
                    self.predVelocityToPlot = self.velocityModel.outputBuffer
                    predStringToPlot = stringModel.outputBuffer

            return predStringToPlot, predOneHotStringToPlot, filteredPrediction, prediction_OneHot


    def stream_thread_new(self):
        self.t=threading.Thread(target=self.stream_analysischunk) #stream_readchunk)
        self.t.start()

    def stream_start(self):
        """adds data to self.data until termination signal"""
        self.initiate()
        print(" -- starting stream")
        self.keepRecording=True # set this to False later to terminate stream
        self.audioToPlot=None # will fill up with threaded recording data
        self.fft=None
        self.dataFiltered=None #same
        self.stream=self.p.open(format=pyaudio.paInt16, channels=1, input_device_index=self.device,
                                rate=self.rate, input=True, frames_per_buffer=self.chunk)
        self.stream_thread_new()

    def list_devices(self):
        # List all audio input devices
        p = pyaudio.PyAudio()
        i = 0
        n = p.get_device_count()
        count = 0
        while i < n:
            dev = p.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                print(str(count) + '. ' + dev['name'])
                count = count + 1
            i += 1

if __name__=="__main__":
    ear=ListenViolin()
    ear.stream_start() #goes forever
    while True:
        print(ear.audioToPlot)
        time.sleep(.1)
    print("DONE")
    ear.keepRecording=False
    ear.stream.close()
    ear.p.terminate()
