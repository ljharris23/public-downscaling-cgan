import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.layers.merge import _Merge


def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(var_list, grads)]


class GradientPenalty(Layer):
    def __init__(self, **kwargs):
        super(GradientPenalty, self).__init__(**kwargs)

    def call(self, inputs):
        target, wrt = inputs
        grad = _compute_gradients(target, [wrt])[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))-1

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)


class RandomWeightedAverage(_Merge):
    def build(self, input_shape):
        super(RandomWeightedAverage, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on exactly 2 inputs')

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on exactly 2 inputs')

        (x, y) = inputs
        shape = K.shape(x)
        weights = K.random_uniform(shape[:1], 0, 1)
        for i in range(len(K.int_shape(x))-1):
            weights = K.expand_dims(weights, -1)
        return x*weights + y*(1-weights)


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (
            s[0],
            None if s[1] is None else s[1]+2*self.padding[0],
            None if s[2] is None else s[2]+2*self.padding[1],
            s[3]
        )

    def call(self, x):
        (i_pad, j_pad) = self.padding
        return tf.pad(x, [[0, 0], [i_pad, i_pad], [j_pad, j_pad], [0, 0]], 'REFLECT')


class SymmetricPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(SymmetricPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (
            s[0],
            None if s[1] is None else s[1]+2*self.padding[0],
            None if s[2] is None else s[2]+2*self.padding[1],
            s[3]
        )

    def call(self, x):
        (i_pad, j_pad) = self.padding
        return tf.pad(x, [[0, 0], [i_pad, i_pad], [j_pad, j_pad], [0, 0]], 'SYMMETRIC')