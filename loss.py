import tensorflow as tf
import numpy as np
import keras.backend as K 

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

def huber_loss(y_true, y_pred, clip_value):
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))

     
def clipped_masked_error(args):
    y_true, y_pred, mask = args
    loss = huber_loss(y_true, y_pred, 1)
    loss *= mask  # apply element-wise mask
    return K.sum(loss, axis=-1)





