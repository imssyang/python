"""Darknet19 Model Defined in Keras."""
import functools
from functools import partial
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from ..utils import compose

# Partial wrapper for Convolution2D with static default argument.
_DarknetConv2D = partial(Conv2D, padding='same')


@functools.wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet weight regularizer for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs.update(kwargs)
    return _DarknetConv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


class Darknet19:
    def __init__(self, inputs):
        """Generate Darknet-19 model for Imagenet classification."""
        body = self.first18_layers(inputs)
        logits = DarknetConv2D(1000, (1, 1), activation='softmax')(body)
        self.model = Model(inputs, logits)

    @classmethod
    def first18_layers(cls):
        """Generate first 18 conv layers of Darknet-19."""
        return compose(
            DarknetConv2D_BN_Leaky(32, (3, 3)),
            MaxPooling2D(),
            DarknetConv2D_BN_Leaky(64, (3, 3)),
            MaxPooling2D(),
            cls.bottleneck_block(128, 64),
            MaxPooling2D(),
            cls.bottleneck_block(256, 128),
            MaxPooling2D(),
            cls.bottleneck_x2_block(512, 256),
            MaxPooling2D(),
            cls.bottleneck_x2_block(1024, 512),
        )

    @classmethod
    def bottleneck_block(cls, outer_filters, bottleneck_filters):
        """Bottleneck block of 3x3, 1x1, 3x3 convolutions."""
        return compose(
            DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
            DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
        )

    @classmethod
    def bottleneck_x2_block(cls, outer_filters, bottleneck_filters):
        """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""
        return compose(
            cls.bottleneck_block(outer_filters, bottleneck_filters),
            DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
        )
