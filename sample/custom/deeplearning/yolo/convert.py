#! /usr/bin/env python
"""
Reads Darknet19 config and weights and creates Keras model with TF backend.

Currently only supports layers in Darknet19 config.

python convert.py cfg_path weights_path h5_path

# 模型转换
执行命令 python convert.py config_path weights_path output_path  
例如：python convert.py model_data/yolov2.cfg model_data/yolov2.weights model_data/yolov2.h5  
config_path：yolov2的配置文件路径  
weights_path：yolov2的权重文件路径  
output_path：输出keras的h5文件  
"""
from dataclasses import dataclass, field
from typing import Any
from typing import List
from typing import Dict, Union
from configparser import ConfigParser
from collections import defaultdict
import io
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv2D, GlobalAveragePooling2D, Input, Lambda, MaxPooling2D, LeakyReLU,
    concatenate, BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model as plot


@dataclass
class DarknetSection:
    layout: int
    name: str
    items: Dict[str, str]

    def __getattr__(self, name):
        if name in self.items:
            return self.items[name]
        return super().__getattr__(name)

    def __repr__(self):
        return str(self.items)


@dataclass
class DarknetConfig:
    path: str
    sections: Dict[str, DarknetSection] = field(default_factory=dict)

    def load(self) -> ConfigParser:
        print(f'Parsing darknet config: {self.path}')
        self._check_with_assert(self.path)
        stream = self._unique_sections(self.path)
        parser = ConfigParser()
        parser.read_file(stream)
        for section in parser.sections():
            name, layout = section.split('_')
            self.sections[section] = DarknetSection(
                int(layout),
                name,
                dict(parser.items(section)),
            )
        return self

    def _check_with_assert(self, path):
        assert path.endswith('.cfg'), f'{path} is not a .cfg file'

    def _unique_sections(self, path) -> io.StringIO:
        """Convert all config sections to have unique names.

        Adds unique suffixes to config sections for compability with configparser.
        """
        section_counters = defaultdict(int)
        output_stream = io.StringIO()
        with open(path) as f:
            for line in f:
                if line.startswith('['):
                    section = line.strip().strip('[]')
                    _section = section + '_' + str(section_counters[section])
                    section_counters[section] += 1
                    line = line.replace(section, _section)
                output_stream.write(line)
        output_stream.seek(0)
        return output_stream


@dataclass
class DarknetConv:
    file: io.FileIO
    section: DarknetSection
    channels: int
    k_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    bn_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    bias: np.ndarray = field(default_factory=lambda: np.array([]))
    filters: int = 0

    def load(self):
        # Darknet serializes convolutional weights as:
        # [bias/beta, [gamma, mean, variance], conv_weights]
        enable_batch_normalize = 'batch_normalize' in self.section.items
        filters = int(self.section.filters)
        size = int(self.section.size)

        self.bias = np.ndarray(
            shape=(filters, ),
            dtype='float32',
            buffer=self.file.read(filters * 4),
        )
        self.filters += filters

        if enable_batch_normalize:
            self.bn_weights = np.ndarray(
                shape=(3, filters),
                dtype='float32',
                buffer=self.file.read(filters * 12),
            )
            self.filters += 3 * filters

        weights_shape = (size, size, self.channels, filters)
        weights_size = np.product(weights_shape)
        self.k_weights = np.ndarray(
            shape=(filters, self.channels, size, size),
            dtype='float32',
            buffer=self.file.read(weights_size * 4),
        )
        self.filters += weights_size
        return self


@dataclass
class DarknetWeights:
    path: str
    file: io.FileIO = field(default=None)
    header: np.ndarray = field(default_factory=lambda: np.array([]))
    #sections: Dict[str, Union[DarknetConv]] = field(default_factory=dict)

    def load_head(self):
        print(f'Loading darknet weights: {self.path}')
        self._check_with_assert(self.path)
        self.file = open(self.path, 'rb')
        self.header = np.ndarray(shape=(4, ), dtype='int32', buffer=self.file.read(16))
        print('Weights Header: ', self.header)
        return self

    def load_remaining(self):
        # Check to see if all weights have been read.
        remaining_filters = len(self.file.read()) / 4
        if remaining_weights > 0:
            print(f'Warning: {remaining_filters} unused weights')
        all_filters = load_filters + remaining_filters
        print(f'Weights read {load_filters} of {all_filters} from Darknet.')
        self.file.close()

    def _check_with_assert(self, path):
        assert path.endswith('.weights'), '{path} is not a .weights file'


class YoloMode:

    def create(self):
        print('Load pretrain YOLO network.')
        config = DarknetConfig('models/yolov2.cfg').load()
        weights = DarknetWeights('models/yolov2.weights').load_head()
        output_path = os.path.expanduser('models/yolov2.h5')
        output_root = os.path.splitext(output_path)[0]

        print('Creating Keras model.')
        net_0 = config.sections['net_0']
        image_height = int(net_0.height)
        image_width = int(net_0.width)
        weight_decay = float(net_0.decay) if 'net_0' in config.sections else 5e-4

        load_filters = 0
        prev_layer = Input(shape=(image_height, image_width, 3))
        all_layers = [prev_layer]
        for secname, section in config.sections.items():
            print(f'Parsing section {secname}')
            if secname.startswith('convolutional'):
                filters = int(section.filters)
                size = int(section.size)
                stride = int(section.stride)
                pad = int(section.pad)
                activation = section.activation
                enable_batch_normalize = 'batch_normalize' in section.items

                prev_layer_shape = K.int_shape(prev_layer)
                prev_channels = prev_layer_shape[-1]
                conv = DarknetConv(weights.file, section, prev_channels).load()
                # DarkNet conv_weights are serialized:
                # Caffe: (out_dim, in_dim, height, width) -> Tensorflow: (height, width, in_dim, out_dim)
                k_weights = np.transpose(conv.k_weights, [2, 3, 1, 0])
                load_filters += conv.filters
                print('conv2d', 'bn' if enable_batch_normalize else '  ', activation, conv.k_weights.shape)

                # Create Conv2D layer
                conv_weights = [k_weights] if enable_batch_normalize else [k_weights, conv.bias]
                conv_padding = 'same' if pad == 1 else 'valid'
                conv_activation = None
                if activation == 'leaky':
                    pass  # Add advanced activation later.
                elif activation != 'linear':
                    raise ValueError(f'Unknown activation function `{activation}` in section {section}')
                conv_layer = (Conv2D(
                    filters, (size, size),
                    strides=(stride, stride),
                    kernel_regularizer=l2(weight_decay),
                    use_bias=not enable_batch_normalize,
                    weights=conv_weights,
                    activation=conv_activation,
                    padding=conv_padding)
                )(prev_layer)

                if enable_batch_normalize:
                    bn_weight_list = [
                        conv.bn_weights[0],  # scale gamma
                        conv.bias,           # shift beta
                        conv.bn_weights[1],  # running mean
                        conv.bn_weights[2]   # running var
                    ]
                    conv_layer = (BatchNormalization(
                        weights=bn_weight_list)
                    )(conv_layer)
                prev_layer = conv_layer

                if activation == 'linear':
                    all_layers.append(prev_layer)
                elif activation == 'leaky':
                    act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                    prev_layer = act_layer
                    all_layers.append(act_layer)

            elif secname.startswith('maxpool'):
                size = int(section.size)
                stride = int(section.stride)
                all_layers.append(MaxPooling2D(
                    padding='same',
                    pool_size=(size, size),
                    strides=(stride, stride))(prev_layer)
                )
                prev_layer = all_layers[-1]

            elif secname.startswith('avgpool'):
                all_layers.append(GlobalAveragePooling2D()(prev_layer))
                prev_layer = all_layers[-1]

            elif secname.startswith('route'):
                ids = [int(i) for i in section.layers.split(',')]
                layers = [all_layers[i] for i in ids]
                if len(layers) > 1:
                    print('Concatenating route layers:', layers)
                    concatenate_layer = concatenate(layers)
                    all_layers.append(concatenate_layer)
                    prev_layer = concatenate_layer
                else:
                    skip_layer = layers[0]  # only one layer to route
                    all_layers.append(skip_layer)
                    prev_layer = skip_layer

            elif secname.startswith('reorg'):
                block_size = int(section.stride)
                assert block_size == 2, 'Only reorg with stride 2 supported.'
                all_layers.append(
                    Lambda(
                        self.space_to_depth_x2,
                        output_shape=self.space_to_depth_x2_output_shape,
                        name='space_to_depth_x2')(prev_layer))
                prev_layer = all_layers[-1]

            elif secname.startswith('region'):
                with open(f'{output_root}_anchors.txt', 'w') as f:
                    print(section.anchors, file=f)

            elif secname.startswith('net') or \
                secname.startswith('cost') or \
                secname.startswith('softmax'):
                pass  # Configs not currently handled during model definition.
            else:
                raise ValueError(f'Unsupported section header type: {secname}')

        # Create and save model.
        model = Model(inputs=all_layers[0], outputs=all_layers[-1])
        model.save(output_path)
        print(model.summary())
        print(f'Saved Keras model to {output_path}')
        plot(model, to_file=f'{output_root}.png', show_shapes=True)
        print(f'Saved model plot to {output_root}.png')



    @staticmethod
    def space_to_depth_x2(x):
        """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
        # Import currently required to make Lambda work.
        # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
        return tf.nn.space_to_depth(x, block_size=2)

    @staticmethod
    def space_to_depth_x2_output_shape(input_shape):
        """Determine space_to_depth output shape for block_size=2.

        Note: For Lambda with TensorFlow backend, output shape may not be needed.
        """
        return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 * input_shape[3]) \
                if input_shape[1] else (input_shape[0], None, None, 4 * input_shape[3])


if __name__ == '__main__':
    YoloMode().create()

