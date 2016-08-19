from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Convolution2D, Flatten, MaxPooling2D
import numpy as np

num_actions = 1
learningRate = 0.001
video_width = 300
video_height = 150

class Policy:
    def __init__(self):
        self.model = self.createModel('relu', learningRate)

    def createModel(self, activationType, learningRate):
            model = Sequential()
            model.add(Convolution2D(8, 3, 3, input_shape=(1, video_width, video_height)))
            model.add(Activation(activationType))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Convolution2D(4, 3, 3))
            model.add(Activation(activationType))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(num_actions, init='lecun_uniform'))
            model.add(Activation("linear"))
            optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
            model.compile(loss="mse", optimizer=optimizer)
            #print model.summary()
            return model
    def getAction(self, state):
        a = self.model.predict(state.reshape(1, 1, video_height, video_width))[0][0]
        a = max(a, -0.1)
        a = min(a, 0.1)
        return a
    def trainModel(self, batch, reward):
        X_batch = np.empty((0, video_height, video_width), dtype = np.float64)
        Y_batch = np.empty((0, num_actions), dtype = np.float64)
        for i in range(len(batch)):
            sample = batch[i]
            state = sample[0]
            action = sample[1]
            X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
            Y_sample = action * reward
            Y_batch = np.append(Y_batch, np.array([[Y_sample]]), axis=0)
        n_train, height, width = X_batch.shape
        #print self.model.get_weights()
        return self.model.train_on_batch(X_batch.reshape(n_train, 1, height, width), Y_batch)