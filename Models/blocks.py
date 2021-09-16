# modified code from https://github.com/MrGiovanni/UNetPlusPlus/blob/master/keras/segmentation_models/xnet/blocks.py
import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Concatenate, Activation, Conv2D
from tensorflow.keras.layers import Add, Multiply, BatchNormalization

def handle_block_names(stage, cols, type_='decoder', type_act='relu'):
    conv_name = '{}_stage{}-{}_conv'.format(type_, stage, cols)
    bn_name = '{}_stage{}-{}_bn'.format(type_, stage, cols)
    act_name = '{}_stage{}-{}_relu'.format(type_, stage, cols)
    up_name = '{}_stage{}-{}_upat'.format(type_, stage, cols)
    add_name = '{}_stage{}-{}_add'.format(type_, stage, cols)
    sigmoid_name = '{}_stage{}-{}_sigmoid'.format(type_, stage, cols)
    mul_name = '{}_stage{}-{}_mul'.format(type_, stage, cols)
    merge_name = 'merge_{}-{}'.format(stage, cols)

    return conv_name, bn_name, act_name, up_name, merge_name, add_name, sigmoid_name, mul_name

def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv',
             bn_name='bn', act_name='relu', act_function='relu'):

    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name,
                   use_bias=not(use_batchnorm)) (x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name) (x)
        x = Activation('relu', name=act_name) (x)

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
                     act_name=relu_name + '1') (x)

        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2',
                     act_name=relu_name + '2') (x)

        return x
    return layer


def Transpose2D_block(filters, stage, cols, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), use_batchnorm=False, skip=None):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name, merge_name,_,_,_ = handle_block_names(stage, cols, type_='decoder')

        x = Conv2DTranspose(filters, transpose_kernel_size, padding='same', name=up_name,
                            strides=upsample_rate, use_bias=not(use_batchnorm)) (input_tensor)
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
                     act_name=relu_name + '2') (x)

        return x
    return layer

def attention_block(filters, skip, stage, cols, upsample_rate=(2,2)):

    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name,_, add_name, sigmoid_name, mul_name = handle_block_names(stage, cols, type_='attention')

        x_up = UpSampling2D(size=upsample_rate, name=up_name+'_before') (input_tensor)
        x_up = ConvRelu(filters, kernel_size=3, conv_name=conv_name+'_before', bn_name=bn_name+'_before', act_name=relu_name+'_before') (x_up)

        x1 = Conv2D(filters, kernel_size=1, padding='same',
                    name=conv_name+'_skip') (skip)
        x1 = BatchNormalization(name=bn_name+'1') (x1)
        x2 = Conv2D(filters, kernel_size=1, padding='same',
                    name=conv_name+'_up') (x_up)
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

def conv_block(filters, stage, cols, kernel_size=3, use_batchnorm=True,
               amount=3, type_act='relu', type_block='encoder'):

    def layer(x):
        act_function = tf.identity if type_act == 'identity' else type_act
        conv_name, bn_name, act_name, _, _, _, _, _ = handle_block_names(stage, cols, type_=type_block, type_act=type_act)
        for i in range(amount):
            temp = '_'+str(i+1)
            x = ConvRelu(filters, kernel_size=kernel_size, use_batchnorm=use_batchnorm,
                          conv_name=conv_name+temp, bn_name=bn_name+temp,
                          act_name=act_name+temp, act_function=act_function) (x)
        return x
    return layer

def z_mu_sigma(filters, stage, cols, use_batchnorm=True, type_block='z'):
    def layer(x):
        mu = conv_block(filters, stage, cols, use_batchnorm=use_batchnorm, amount=1,
                        kernel_size=1, type_act='identity', type_block='mu') (x)
        sigma = conv_block(filters, stage, cols, use_batchnorm=use_batchnorm, amount=1,
                           kernel_size=1, type_act='softplus', type_block='sigma') (x)

        z = Multiply(name='z_stage{}-{}_mul'.format(stage,cols)) ([
            sigma, tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)])
        z = Add(name='z_stage{}-{}_add'.format(stage,cols)) ([mu, z])
        return z, mu, sigma
    return layer
