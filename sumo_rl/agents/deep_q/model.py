import os
# from typing_extensions import Self
from pandas import wide_to_long
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
"""pip install 'h5py==2.10.0' --force-reinstall"""

import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras import backend as K
from pathlib import Path

class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim, opt, retrain=None, 
    path=None, build=True):
        self._input_dim = input_dim
        self._num_layers = num_layers
        self._width = width
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._opt = opt
        if build:
            self._model = self._build_model(num_layers, width, opt, retrain, path)
        else:
            self._model = None

    def change_lr(self,decay,change=False):
        new_lr = self._learning_rate*decay
        if change:
            if new_lr < 0.00001:
                learning_rate = 0.00001
            else:
                learning_rate = new_lr
            self._learning_rate = learning_rate
            K.set_value(self._model.optimizer.learning_rate, learning_rate)
            

    def clone(self):
        instance = TrainModel(num_layers=self._num_layers, width=self._width, batch_size=self._batch_size, 
        learning_rate=self._learning_rate, input_dim=self._input_dim, output_dim=self._output_dim, opt=self._opt, build=False)
        instance._model = clone_model(self._model)
        return instance

    def _build_model(self, num_layers, width, opt, retrain, path, moment = 0.9):
        """
        Build and compile a fully connected deep neural network
        """
        if retrain:
            model = load_model(path)
            # self._model.save(os.path.join(path, 'trained_model_epi{}.h5'.format(epi)))
        else:
            inputs = keras.Input(shape=(self._input_dim,))
            x = layers.Dense(width, activation='relu')(inputs)
            for _ in range(num_layers):
                x = layers.Dense(width, activation='relu')(x)
            outputs = layers.Dense(self._output_dim)(x)
            model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')

        if opt == "adam":
            model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        elif opt == "rmsprop":
            model.compile(loss=losses.mean_squared_error, optimizer=RMSprop(lr=self._learning_rate))
        elif opt == "sgd":
            model.compile(loss=losses.mean_squared_error, optimizer=SGD(lr=self._learning_rate, momentum=moment))
        
        return model
    
    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state,verbose=0)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states, verbose=0)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)
        """loss = self._model.train_on_batch(states, q_sa)
        print("loss", loss)"""

    def save_model(self, path, epi):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model_epi{}.h5'.format(epi)))
        # plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path, epi):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path, epi)


    def _load_my_model(self, model_folder_path, epi):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = str(Path(os.path.join(model_folder_path, 'trained_model_epi{}.h5'.format(epi))))
        print("model_file_path",model_file_path)
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim