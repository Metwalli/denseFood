from __future__ import print_function
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Input, BatchNormalization, ZeroPadding2D, \
    Activation, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Concatenate, GlobalMaxPooling2D
from keras.models import Model
from keras.activations import elu
from keras.backend import image_data_format, int_shape
from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.utils import plot_model


channel_axis = 3 if image_data_format() == 'channels_last' else 1
eps = 1.001e-5
num_classes = 10

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def conv_layer(x, num_filters, kernel, stride=1, padding='same', layer_name="conv"):
    conv = Conv2D(num_filters,
                  kernel_size=kernel,
                  use_bias=False,
                  strides=stride,
                  padding=padding,
                  name=layer_name)(x)
    return conv


def Global_Average_Pooling(x, stride=1, name=None):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return GlobalAveragePooling2D(name=name)(x)
    # But maybe you need to install h5py and curses or not


def Average_pooling(x, pool_size=[2, 2], stride=2, name=None):
    return AveragePooling2D(pool_size, strides=stride, name=name)(x)


def Max_pooling(x, pool_size=[3, 3], stride=2, padding='SAME', name=None):
    return MaxPooling2D(pool_size=pool_size, strides=stride, padding=padding, name=name)(x)


def activation_fn(x, name=None):
    return Activation('relu', name=name)(x)


def batch_normalization_fn(x, name=None):
    return BatchNormalization(axis=channel_axis, epsilon=eps, name=name)(x)

def dropout_fn(x, rate):
    return Dropout(rate=rate)(x)

def dense_fn(layer, filters=100):
    return Dense(filters)(layer)

def classifier_fn(layer, num_labels=2, actv='softmax'):
    return Dense(num_labels, activation=actv)(layer)

def concat_fn(layers, axis=channel_axis, name=None):
    return Concatenate(axis=axis, name=name)(layers)

def load_densenet_model(use_weights, pooling='avg'):
    weights = 'imagenet' if use_weights == True else None
    base_model = DenseNet121(include_top=False, weights=weights, input_tensor=Input(shape=(224, 224, 3)),
                             input_shape=(224, 224, 3), pooling=pooling)
    return base_model

def load_inceptionv3_model(use_weights, pooling='avg', input_tensor=None):
    weights = 'imagenet' if use_weights == True else None
    base_model = InceptionV3(include_top=False, weights=weights, input_tensor=input_tensor,
                             input_shape=(299, 299, 3), pooling=pooling)
    return base_model
def load_VGG_model(use_weights, pooling=None, input_tensor=None):
    weights = 'imagenet' if use_weights == True else None
    base_model = VGG16(include_top=True, weights=weights, input_tensor=input_tensor,
                             input_shape=(224, 224, 3), pooling=pooling)
    return base_model
def load_ResNet_model(use_weights, pooling='avg', input_tensor=None):
    weights = 'imagenet' if use_weights == True else None
    base_model = ResNet50(include_top=False, weights=weights, input_tensor=input_tensor,
                             input_shape=(224, 224, 3), pooling=pooling)
    return base_model

# Inception v3 Model
class Inceptionv3Model():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        base_model = load_inceptionv3_model(self.use_imagenet_weights, pooling='avg')
        out = base_model.layers[-1].output
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=base_model.input, outputs=classifier)
        return model

# VGG19 Base Model
class VGGModel():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        base_model = load_VGG_model(self.use_imagenet_weights)
        out = base_model.layers[-2].output
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=base_model.input, outputs=classifier)
        return model

# ResNet50 Base Model
class RestNetModel():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        base_model = load_ResNet_model(self.use_imagenet_weights)
        out = base_model.layers[-1].output
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=base_model.input, outputs=classifier)
        return model

# DenseNet121 Base Model
class DenseNetBaseModel():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        base_model = load_densenet_model(self.use_imagenet_weights)
        out = base_model.layers[-1].output
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')
        model = Model(inputs=base_model.input, outputs=classifier)
        return model

# Densenet Modify

class DenseNet121_Modify():
    def __init__(self, num_labels, use_imagenet_weights=True):
        self.num_labels = num_labels
        self.use_imagenet_weights = use_imagenet_weights
        self.model = self.get_model()

    def get_model(self):
        base_model = load_densenet_model(self.use_imagenet_weights)
        model_out = base_model.layers[-1].output
        out = dense_fn(model_out, 2048)
        out = dropout_fn(out, 0.5)
        # concat = concat_fn([block2_out, block3_out, model_out], axis=1, name='Concatblocks234')
        classifier = classifier_fn(layer=out, num_labels=self.num_labels, actv='softmax')

        model = Model(inputs=base_model.input, outputs=[classifier])
        return model


class DenseFoodModel():
    def __init__(self, num_labels, num_layers_per_block):
        self.num_labels = num_labels
        self.model = self.DenseNet(num_layers_per_block)

    def dense_block(self, x, blocks, name):
        """A dense block.

        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        for i in range(blocks):
            x = self.conv_block(x, 32, name=name + '_block' + str(i + 1))
        return x

    def transition_block(self, x, reduction, name):
        """A transition block.

        # Arguments
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        bn_axis = 3 if image_data_format() == 'channels_last' else 1
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name + '_bn')(x)
        x = Activation('elu', name=name + '_elu')(x)
        x = Conv2D(int(int_shape(x)[bn_axis] * reduction), 1,
                          use_bias=False,
                          name=name + '_conv')(x)
        x = MaxPooling2D(2, strides=2, name=name + '_pool')(x)
        return x

    def conv_block(self, x, growth_rate, name):
        """A building block for a dense block.

        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.

        # Returns
            Output tensor for the block.
        """
        bn_axis = 3 if image_data_format() == 'channels_last' else 1
        x1 = BatchNormalization(axis=bn_axis,
                                       epsilon=1.001e-5,
                                       name=name + '_0_bn')(x)
        x1 = Activation('elu', name=name + '_0_elu')(x1)
        x1 = Conv2D(4 * growth_rate, 1,
                           use_bias=False,
                           name=name + '_1_conv')(x1)
        x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       name=name + '_1_bn')(x1)
        x1 = Activation('elu', name=name + '_1_elu')(x1)
        x1 = Conv2D(growth_rate, 3,
                           padding='same',
                           use_bias=False,
                           name=name + '_2_conv')(x1)
        x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x

    def DenseNet(self, blocks, input_shape=(224, 224, 3)):

        img_input = Input(shape=input_shape)

        bn_axis = 3 if image_data_format() == 'channels_last' else 1

        x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
        x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
        x = BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
        x = Activation('elu', name='conv1/elu')(x)
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = MaxPooling2D(3, strides=2, name='pool1')(x)

        x = self.dense_block(x, blocks[0], name='conv2')
        x = self.transition_block(x, 0.5, name='pool2')
        x = self.dense_block(x, blocks[1], name='conv3')
        x = self.transition_block(x, 0.5, name='pool3')
        x = self.dense_block(x, blocks[2], name='conv4')
        x = self.transition_block(x, 0.5, name='pool4')
        x = self.dense_block(x, blocks[3], name='conv5')

        x = BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
        x = Activation('elu', name='elu')(x)


        x = GlobalAveragePooling2D(name='avg_pool_')(x)


        x = classifier_fn(x, self.num_labels, actv='softmax')

        model = Model(img_input, x, name='densenet')

        return model

