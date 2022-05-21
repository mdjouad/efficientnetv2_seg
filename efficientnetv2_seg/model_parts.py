from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.activations import swish


def res_block(x, filters):
    shortcut = layers.Conv2D(filters, kernel_size=1, padding="same", use_bias=False)(x)

    x = layers.Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = swish(x)

    x = layers.Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, shortcut])
    x = swish(x)

    return x


def attention_gating_block(g, x, filters):

    x_ = layers.Conv2D(filters, kernel_size=1, use_bias=False)(x)
    x_ = layers.BatchNormalization()(x_)

    g = layers.Conv2D(filters, kernel_size=1, use_bias=False)(g)
    g = layers.BatchNormalization()(g)

    psi = layers.add([g, x_])
    psi = swish(psi)

    psi = layers.Conv2D(1, kernel_size=1)(psi)
    psi = layers.BatchNormalization()(psi)
    psi = layers.Activation("sigmoid")(psi)

    att = layers.multiply([psi, x])

    return att


def decoder_block(x, skip, filters, use_attention=False):
    x = layers.UpSampling2D(size=2)(x)
    
    if use_attention is True:
        skip = attention_gating_block(g=x, x=skip, filters=filters)

    x = layers.concatenate([x, skip], axis=-1)
    x = res_block(x, filters)

    return x


def model_head(x, num_classes, activation):
    assert activation in ("sigmoid", "softmax"), "Unknow {activation} activation"

    x = layers.Conv2D(num_classes, kernel_size=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    return x
