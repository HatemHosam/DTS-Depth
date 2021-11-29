import tensorflow as tf
from tensorflow.keras.models import Model


# DTS-Model definition
def DTS_model( shape = (None,None,3))
    base = tf.keras.applications.MobileNetV2(input_shape = shape, weights= 'imagenet', include_top=False, pooling= None)
    out1 = tf.Conv2D(1024, (1,1), activation = 'linear')(base.output)
    out_f = tf.nn.depth_to_space(out1, 32)
    model = Model(base.input, out_f)
    return model
