from tensorflow.keras.layers import Layer, Add, Conv2D, LeakyReLU, BatchNormalization, AveragePooling2D

from layers import ReflectionPadding2D, SymmetricPadding2D


class Conv2DPadding(Layer):
    def __init__(self, filters, kernel_size, stride, padding, dilations):
        super(Conv2DPadding, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilations
        if not isinstance(dilations, int):
            # padding calculation in build() would need to be adjusted to handle a tuple/list
            raise NotImplementedError("Only integer dilation is supported.")
        if padding is None:
            raise ValueError("padding should not be None")

    def build(self, x):
        if self.padding in ('reflect', 'symmetric'):
            pad = tuple((self.dilation*(s-1))//2 for s in self.kernel_size)  # only works if s is odd, or dilation is even
            if self.padding == 'reflect':
                self.padref = ReflectionPadding2D(padding=pad)
            elif self.padding == 'symmetric':
                self.symref = SymmetricPadding2D(padding=pad)
            self.convval = Conv2D(filters=self.filters,
                                  kernel_size=self.kernel_size,
                                  strides=(self.stride, self.stride),
                                  padding='valid',
                                  dilation_rate=self.dilation)
        else:
            self.convsam = Conv2D(filters=self.filters,
                                  kernel_size=self.kernel_size,
                                  strides=(self.stride, self.stride),
                                  padding='same',
                                  dilation_rate=self.dilation)

    def call(self, x):
        if self.padding in ('reflect', 'symmetric'):
            if self.padding == 'reflect':
                x = self.padref(x)
            elif self.padding == 'symmetric':
                x = self.symref(x)
            return self.convval(x)
        else:  # same
            return self.convsam(x)


def residual_block(x, filters, conv_size=(3, 3), stride=1, dilations=1, relu_alpha=0.2, norm=None, padding=None, force_1d_conv=False):
    in_channels = int(x.shape[-1])
    x_in = x

    if stride > 1:
        x_in = AveragePooling2D(pool_size=(stride, stride))(x_in)
    if force_1d_conv or (filters != in_channels):
        x_in = Conv2D(filters=filters, kernel_size=(1, 1))(x_in)

    # first block of activation and 3x3 convolution (possibly strided, although we don't use this)
    x = LeakyReLU(relu_alpha)(x)
    x = Conv2DPadding(filters=filters, kernel_size=conv_size, stride=stride, dilations=dilations, padding=padding)(x)
    if norm == "batch":
        x = BatchNormalization()(x)
    elif norm is None:
        pass
    else:
        print("norm type not implemented")

    # second block of activation and 3x3 unstrided convolution
    x = LeakyReLU(relu_alpha)(x)
    x = Conv2DPadding(filters=filters, kernel_size=conv_size, stride=1, dilations=dilations, padding=padding)(x)
    if norm == "batch":
        x = BatchNormalization()(x)
    elif norm is None:
        pass
    else:
        print("norm type not implemented")

    # skip connection
    x = Add()([x, x_in])

    return x


def const_upscale_block(const_input, steps, filters):
    # Map (N x kH x kW x C) to (N x H x W x f), where k is downscaling factor
    const_output = const_input
    for step in steps:
        const_output = Conv2D(filters=filters, kernel_size=(step, step), strides=step, padding="valid", activation="relu")(const_output)
    return const_output
