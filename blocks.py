# modified code from https://github.com/MrGiovanni/UNetPlusPlus/blob/master/keras/segmentation_models/xnet/blocks.py

from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, Conv2D
from tensorflow.keras.layers import Add, Multiply, BatchNormalization
from tensorflow.keras.layers import Concatenate, Activation

def handle_block_names(stage, cols, type_='decoder'):
    temp = 'upsample' if type_ == 'decoder' or type_ == 'attention' else 'downsample'
    conv_name = '{}_stage{}-{}_conv'.format(type_, stage, cols)
    bn_name = '{}_stage{}-{}_bn'.format(type_, stage, cols)
    relu_name = '{}_stage{}-{}_relu'.format(type_, stage, cols)
    up_name = '{}_stage{}-{}_{}'.format(type_, stage, cols, temp)
    add_name = '{}_stage{}-{}_add'.format(type_, stage, cols)
    sigmoid_name = '{}_stage{}-{}_sigmoid'.format(type_, stage, cols)
    mul_name = '{}_stage{}-{}_mul'.format(type_, stage, cols)
    merge_name = 'merge_{}-{}'.format(stage, cols)

    return conv_name, bn_name, relu_name, up_name, merge_name, add_name, sigmoid_name, mul_name

def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv',
             bn_name='bn', relu_name='relu'):

    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name,
                   use_bias=not(use_batchnorm)) (x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name) (x)
        x = Activation('relu', name=relu_name) (x)

        return x
    return layer

def UpRelu(filters, transpose=False, use_batchnorm=False, conv_name='c_up',
           bn_name='b_up', relu_name='r_up', up_name='upsample', upsample_rate=(2,2)):

    def layer(input_tensor):
        if transpose:
            x = Conv2DTranspose(filters, kernel_size=upsample_rate, padding='same') (input_tensor)
        else:
            x = UpSampling2D(size=upsample_rate, name=up_name) (input_tensor)

        x = ConvRelu(filters, kernel_size=3, use_batchnorm=use_batchnorm,
                     conv_name=conv_name, bn_name=bn_name, relu_name=relu_name) (x)

        return x
    return layer

def Upsample2D_block(filters, stage, cols, kernel_size=(3,3), upsample_rate=(2,2),
                     use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name, merge_name,_,_,_ = handle_block_names(stage, cols, type_='decoder')

        x = UpSampling2D(size=upsample_rate, name=up_name) (input_tensor)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            if type(skip) is list:
                x = Concatenate(name=merge_name) ([x] + skip)
            else:
                x = Concatenate(name=merge_name) ([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1',
                     relu_name=relu_name + '1') (x)

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2',
                     relu_name=relu_name + '2') (x)

        return x
    return layer


def Transpose2D_block(filters, stage, cols, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name, merge_name,_,_,_ = handle_block_names(stage, cols, type_='decoder')

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm)) (input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name+'1') (x)
        x = Activation('relu', name=relu_name+'1') (x)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            # print("\nskip = {}".format(skip))
            if type(skip) is list:
                merge_list = []
                merge_list.append(x)
                for l in skip:
                    merge_list.append(l)
                x = Concatenate(name=merge_name) (merge_list)

            else:
                x = Concatenate(name=merge_name) ([x, skip])

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2',
                     relu_name=relu_name + '2') (x)

        return x
    return layer

def down_block(filters, stage, cols, kernel_size=(3,3), use_batchnorm=False):

    def layer(input_tensor):
        conv_name, bn_name, relu_name,_,_,_,_,_ = handle_block_names(stage, cols, type_='encoder')
        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1',
                     relu_name=relu_name + '1') (input_tensor)

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2',
                     relu_name=relu_name + '2') (x)
        return x
    return layer

def attention_block(filters, skip, stage, cols, upsample_rate=(2,2)):

    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name,_, add_name, sigmoid_name, mul_name = handle_block_names(stage, cols, type_='attention')

        x_up = UpRelu(filters, conv_name=conv_name+'_before', bn_name=bn_name+'_before',
                      relu_name=relu_name+'_before', up_name=up_name+'_before',
                      use_batchnorm=True, upsample_rate=upsample_rate) (input_tensor)

        x1 = Conv2D(filters, kernel_size=1, padding='same', name=conv_name+'_skip') (skip)
        x1 = BatchNormalization(name=bn_name+'1') (x1)
        x2 = Conv2D(filters, kernel_size=1, padding='same', name=conv_name+'_up') (x_up)
        x2 = BatchNormalization(name=bn_name+'2') (x2)

        x = Add(name=add_name) ([x1,x2])
        x = Activation('relu', name=relu_name) (x)
        x = Conv2D(1, kernel_size=1, padding='same', name=conv_name) (x)
        x = BatchNormalization(name=bn_name+'3') (x)
        x = Activation('sigmoid', name=sigmoid_name) (x)
        x = Multiply(name=mul_name) ([skip,x])

        return x
    return layer

def DeepSupervision(classes):

    def layer(seg_branches):
        concat_list = []
        for k,i in enumerate(seg_branches):
            temp = Conv2D(classes, kernel_size=1, padding='same', name='ds_conv_'+str(k+1)) (i)
            temp = Activation('sigmoid', name='ds_sigmoid_'+str(k+1)) (temp)
            concat_list.append(temp)

        return concat_list
    return layer
