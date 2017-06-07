import numpy as np
from sklearn.externals import joblib #save model to disk

class SKitLearnModel(object):
    def __init__(self, path, model_name):
        # Load a previously saved meta graph in the default graph
        self.model_name = model_name
        self.path = path
        self.model = joblib.load(path + model_name + '.pkl')
        # Como averiguar el numero de inputs? outputs?
        # n_classes = self.model.n_classes_  #or classes_.shape[0]
        self.ninputs = self.model.n_features_

        self.inputWinLen = 1
        self.inputBufferSize = 1

        self.inputFrameBuffer = np.ones(shape=(self.inputBufferSize, self.ninputs)) * 0
        self.maxPredValuesToStore = 1000
        self.outputBuffer = np.zeros(self.maxPredValuesToStore)  # [0]*100 #np.zeros(self.predictionWinL)


    def processNextBatch(self, num_new_frames):
        prediction = self.predict_values(np.transpose(self.inputFrameBuffer))
        #print('prediction: ', prediction)
        if len(prediction) < num_new_frames:
            num_new_frames = len(prediction)
        self.outputBuffer = np.hstack((self.outputBuffer, prediction[-num_new_frames:]))  # just add to the buffer the new predictions!!
        self.outputBuffer = self.outputBuffer[num_new_frames:]


    def addFrame(self, inputFrame, keepLength=False):
        self.inputFrameBuffer = np.vstack((self.inputFrameBuffer, inputFrame))
        if keepLength:
            self.inputFrameBuffer = self.inputFrameBuffer[1:, :]

    def predict_values(self, inputBatch):
        numBatches = 1
        if self.inputWinLen == 1:                            #this is for Feed-Forward Nets
            #prediction = np.zeros(inputBatch.shape[1])
            Xs_i = np.zeros(shape=(numBatches, 40))
            for iFrame in range(inputBatch.shape[1]):
                Xs_i[0, :] = inputBatch[:, iFrame].T
                #[_, pred] = self.session.run([self.predictionOP, self.outputTensor], feed_dict={self.input: Xs_i})
                #prediction[iFrame] = int(np.argmax(pred))
            prediction = self.model.predict(Xs_i)

            #prediction = prediction.flatten()
        # if the model output is a softmax for classification, then just return the most probable class.
        # Additionally a confidence value could be computed. For now ...:

        return prediction

