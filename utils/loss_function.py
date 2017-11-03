from keras.objectives import *
from keras.metrics import binary_crossentropy
import keras.backend as K
import tensorflow as tf
import keras
from tensorflow.python.client import device_lib
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import variables as tf_variables

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

# Softmax cross-entropy loss function for pascal voc segmentation
# and models which do not perform softmax.
# tensorlow only
def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    #y_pred = tf.Print(y_pred,[tf.shape(y_pred)],summarize=4)
    log_softmax = tf.nn.log_softmax(y_pred)
    
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    '''
	first index for one hot encoding is the number of indicies
    '''
    #y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1])
    '''
       creates a tensor where the indecies are the last axes, and the tensors at that
       index are the other 3 dimensions
    '''
    #y_true = tf.Print(y_true,[tf.shape(y_true)],message="y_true",summarize=4)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)
    
    

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    #cross_entropy = tf.Print(cross_entropy,[cross_entropy],message="result",summarize=1000)
    cross_entropy_mean = K.mean(cross_entropy)
    #cross_entropy_mean = tf.Print(cross_entropy_mean,[cross_entropy_mean],message="result",summarize=200)
    return cross_entropy_mean


# Softmax cross-entropy loss function for coco segmentation
# and models which expect but do not apply sigmoid on each entry
# tensorlow only
def binary_crossentropy_with_logits(ground_truth, predictions):
    return K.mean(K.binary_crossentropy(ground_truth,
                                        predictions,
                                        from_logits=True),
                  axis=-1)
def flatten_categorical_crossentropy(target, output):
        output = tf.Print(output,[output])
        target = tf.Print(target,[target], summarize=1000)
	
	output /= tf.reduce_sum(output,
                                axis=len(output.get_shape()) - 1,
                                keep_dims=True)
	#output = tf.Print(output,[tf.shape(output)],message="prediction")
        # manual computation of crossentropy
        _epsilon = _to_tensor(1e-08, output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(target * tf.log(output),
		 	axis=len(output.get_shape()) - 1)
        


