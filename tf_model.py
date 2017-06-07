import tensorflow as tf
import numpy as np

class TfModel(object):
    def __init__(self, path, model_name, predictionOP, minValue, inputBufferSize=1, one_hot_encoding=False, do_hysteresis=False):
        # Load a previously saved meta graph in the default graph
        self.one_hot_encoding = one_hot_encoding
        self.one_hot_prediction = []
        self.model_name = model_name
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(path + model_name + '.ckpt.meta')
        # We can now access the default graph where all our metadata has been loaded
        #self.graph = tf.Graph()
        self.graph = tf.get_default_graph()
        #self.graph.as_default()
        self.session = tf.Session()
        # Initialize values with saved data
        saver.restore(self.session, path + model_name + '.ckpt')
        self.input = self.graph.get_tensor_by_name('X:0')
        self.predictionOP = self.graph.get_operation_by_name(predictionOP)
        self.outputTensor = self.graph.get_tensor_by_name(predictionOP + ':0')
        if len(self.input.get_shape().as_list()) > 2:
            self.inputWinLen = self.input.get_shape().as_list()[2]
            self.inputBufferSize = self.inputWinLen
        else:
            self.inputWinLen = 1
            self.inputBufferSize = inputBufferSize
        self.ninputs = self.input.get_shape().as_list()[1]
        self.inputFrameBuffer = np.ones(shape=(self.inputBufferSize, self.ninputs)) * 0
        self.maxPredValuesToStore = 1000
        #self.numPredictedValuesToPlot = 1000
        self.outputBuffer = np.zeros(self.maxPredValuesToStore)  # [0]*100 #np.zeros(self.predictionWinL)
        w, h = self.maxPredValuesToStore, 5
        self.outputBufferOneHot = [[0 for x in range(w)] for y in range(h)]
        self.outputBufferOneHot = np.array(self.outputBufferOneHot)
        #self.norm_max = 120
        #self.norm_offset = 1
        #self.predVelocityToPlot = [0] * self.numPredictedValuesToPlot
        self.do_hysteresis = do_hysteresis
        if do_hysteresis:
            transitionConsolidationL=3
            self.hysteresisbuffer = Hysteresisbuffer(transitionConsolidationL)

    def processNextBatch(self, num_new_frames):
        #first cut inputFrameBuffer as it may be larger that num_new_frames.
        #self.inputFrameBuffer = self.inputFrameBuffer[-num_new_frames:]
        #then run one prediction for each available Batch
        n_predictions = range(0, self.inputFrameBuffer.shape[0]//self.inputWinLen)
        filreredPrediction = []
        prediction_OneHot = []
        for iBatch in n_predictions:
            [prediction, prediction_OneHot] = self.predict_values(np.transpose(self.inputFrameBuffer[0:self.inputWinLen]))
            self.inputFrameBuffer = self.inputFrameBuffer[self.inputWinLen:]

            if self.do_hysteresis:
                # Filter the prediction through the hysteresis buffer.
                self.hysteresisbuffer.push(prediction)
                filreredPrediction = self.hysteresisbuffer.pop()
            else:
                filreredPrediction = prediction

            num_pred_frames = len(filreredPrediction)
            if num_pred_frames: #len(filreredPrediction):
                self.outputBuffer = np.hstack((self.outputBuffer, filreredPrediction))  # just add to the buffer the new predictions!!
                self.outputBuffer = self.outputBuffer[num_pred_frames:]
            if self.one_hot_encoding:
                new_col = np.array(prediction_OneHot)[..., None]
                aux = np.append(self.outputBufferOneHot, new_col, 1)
                aux = aux[:, 1:]
                self.outputBufferOneHot = aux

        return filreredPrediction, prediction_OneHot
        #[prediction, prediction_OneHot] = self.predict_values(np.transpose(self.inputFrameBuffer))
        ##print('prediction: ', prediction)
        #if len(prediction) < num_new_frames:
        #    num_new_frames = len(prediction)
        #self.outputBuffer = np.hstack((self.outputBuffer, prediction[-num_new_frames:]))  # just add to the buffer the new predictions!!
        #self.outputBuffer = self.outputBuffer[num_new_frames:]
        #if self.one_hot_encoding:
        #    new_col = np.array(prediction_OneHot)[..., None]
        #    aux = np.append(self.outputBufferOneHot, new_col, 1)
        #    aux = aux[:, 1:]
        #    self.outputBufferOneHot = aux

    def addFrame(self, inputFrame, keepLength=True):
        self.inputFrameBuffer = np.vstack((self.inputFrameBuffer, inputFrame))
        if keepLength:
            self.inputFrameBuffer = self.inputFrameBuffer[1:, :]

    def predict_values(self, inputBatch):
        numBatches = 1
        one_hot_prediction = []
        if self.inputWinLen == 1:                            #this is for Feed-Forward Nets
            prediction = np.zeros(inputBatch.shape[1])
            Xs_i = np.zeros(shape=(numBatches, self.ninputs))
            for iFrame in range(inputBatch.shape[1]):
                Xs_i[0, :] = (inputBatch[:, iFrame].T / self.norm_max) + self.norm_offset
                [_, pred] = self.session.run([self.predictionOP, self.outputTensor], feed_dict={self.input: Xs_i})
                prediction[iFrame] = int(np.argmax(pred))
                #if prediction[iFrame] > 0:
                #    print("pred:", pred, "-->string:", prediction[iFrame])
        else:
            Xs_i = np.zeros(shape=(numBatches, self.ninputs, self.inputWinLen, 1))   #this is for convolutional nets
            Xs_i[0, :, :, 0] = inputBatch #(inputBatch / self.norm_max) + self.norm_offset
            [_, prediction] = self.session.run([self.predictionOP, self.outputTensor], feed_dict={self.input: Xs_i})
            if self.one_hot_encoding:
                one_hot_prediction = prediction[0] #position 0 of prediction. This is ok if only one value is predicted.
                prediction = np.array(np.argmax(prediction))
                #print(self.one_hot_prediction)

            prediction = prediction.flatten()
        # if the model output is a softmax for classification, then just return the most probable class.
        # Additionally a confidence value could be computed. For now ...:

        return prediction, one_hot_prediction

class Hysteresisbuffer(object):
    def __init__(self, hysteresis_length):
        self.hysteresis_length = hysteresis_length
        self.internal_buffer = np.zeros(hysteresis_length)
        self.inTransition = False
        self.from_state = 0
        self.to_state = 0
        self.new_data_counter = 0
        self.consolidate_counter = 0

    def push(self, new_data):
        #for now: assume new_data is just one value
        new_data = new_data[0]
        if new_data != self.from_state:
            if (self.inTransition and new_data == self.to_state):
                if self.consolidate_counter >= self.hysteresis_length:    #consolidate state
                    self.from_state = self.to_state
                    self.consolidate_counter = 0
                    self.inTransition = False
                    self.internal_buffer = np.hstack((self.internal_buffer, new_data))
                    self.new_data_counter = self.new_data_counter + 1
                else:   #consolidating...
                    self.consolidate_counter = self.consolidate_counter + 1

            else:
                self.inTransition = True
                self.to_state = new_data
                self.consolidate_counter = 1
        else:
            self.consolidate_counter = 0
            self.inTransition = False
            self.internal_buffer = np.hstack((self.internal_buffer, new_data))
            #self.internal_buffer = self.internal_buffer[len(new_data), :]
            self.new_data_counter = self.new_data_counter + 1

    def pop(self):
        toreturn=self.internal_buffer[0:self.new_data_counter]
        self.internal_buffer = self.internal_buffer[self.new_data_counter:]
        return toreturn

class BaggingMododel(object):
    def __init__(self, models):
        self.models = models