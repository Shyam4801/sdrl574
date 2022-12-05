
from config.device_config import device
from config.hyperparameters import *

from keras.models import Sequential, Model, load_model, model_from_config
from keras.layers import Dense, Conv2D, Flatten, Input, concatenate, Lambda
from keras import optimizers
from keras import initializers
import gc
from loss import clipped_masked_error

HUBER_DELTA = 0.5

def clone_model(model, custom_objects={}):
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone


class Hdqn:
    def __init__(self):
        self.enable_dueling_network = False
        # 3 layer Deep Q and target network
        rmsProp = optimizers.RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=1e-08, decay=0.0)
        Q_net = Sequential()
        Q_net.add(Conv2D(32, (8, 8), strides=4, activation='relu', padding='valid', input_shape=(84, 84, 4)))
        Q_net.add(Conv2D(64, (4, 4), strides=2, activation='relu', padding='valid'))
        Q_net.add(Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid'))
        Q_net.add(Flatten())
        Q_net.add(Dense(HIDDEN_NODES, activation='relu',
                                kernel_initializer=initializers.random_normal(stddev=0.01, seed=SEED)))
        Q_net.add(Dense(nb_Action, activation='linear',
                                kernel_initializer=initializers.random_normal(stddev=0.01, seed=SEED)))
        Q_net.compile(loss='mse', optimizer=rmsProp)

        Target = Sequential()
        Target.add(Conv2D(32, (8, 8), strides=4, activation='relu', padding='valid', input_shape=(84, 84, 4)))
        Target.add(Conv2D(64, (4, 4), strides=2, activation='relu', padding='valid'))
        Target.add(Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid'))
        Target.add(Flatten())
        Target.add(Dense(HIDDEN_NODES, activation='relu',
                                    kernel_initializer=initializers.random_normal(stddev=0.01, seed=SEED)))
        Target.add(Dense(nb_Action, activation='linear',
                                    kernel_initializer=initializers.random_normal(stddev=0.01, seed=SEED)))
        Target.compile(loss='mse', optimizer=rmsProp)

        if not self.enable_dueling_network:
            self.controllerNet = Q_net
            self.targetControllerNet = Target

            self.controllerNet.reset_states()
            self.targetControllerNet.set_weights(self.controllerNet.get_weights())
        else:
            layer = Q_net.layers[-2]
            nb_output = Q_net.output._keras_shape[-1]
            y = Dense(nb_output + 1, activation='linear', kernel_initializer=initializers.random_normal(stddev=0.01))(
                layer.output)
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                                 output_shape=(nb_output,))(y)

            self.controllerNet = Model(inputs=Q_net.input, outputs=outputlayer)
            self.controllerNet.compile(optimizer=rmsProp, loss='mse')
            self.targetControllerNet = clone_model(self.controllerNet)
            self.targetControllerNet.compile(optimizer=rmsProp, loss='mse')

    def saveWeight(self, subgoal):
        self.controllerNet.save_weights(recordFolder + '/policy_subgoal_' + str(subgoal) + '.h5')

    def loadWeight(self, subgoal):
        self.controllerNet.load_weights(recordFolder + '/policy_subgoal_' + str(subgoal) + '.h5')
        self.controllerNet.reset_states()

    def clear_memory(self):
        del self.controllerNet
        del self.targetControllerNet
        gc.collect()
