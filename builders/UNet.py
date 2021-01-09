# https://github.com/MrGiovanni/UNetPlusPlus/blob/master/keras/segmentation_models/unet/builder.py

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D

from utils import get_layer_number, to_tuple
from blocks import Transpose2D_block, Upsample2D_block
from blocks import down_block, attention_block

def build_unet(use_backbone, backbone, classes, skip_connection_layers,
               decoder_filters=(256,128,64,32),
               upsample_rates=(2,2,2,2),
               n_upsample_blocks=4,
               block_type='upsampling',
               activation='sigmoid',
               input_shape=(None,None,3),
               use_batchnorm=True,
               attention=False):

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # Using backbone for the encoder
    if use_backbone:
        input = backbone.input
        if 'vgg' not in backbone.name:
            x = backbone.output
        else:
            x = backbone.layers[-2].output

        skip_layers_list = ([get_layer_number(backbone, l) if isinstance(l, str) else l 
                             for l in skip_connection_layers])

    # Using Conv+relu for the encoder
    else:
        encoder_filters = (decoder_filters[0]*2,) + decoder_filters
        input = Input(shape=input_shape)
        downterm = [None] * (n_upsample_blocks+1)

        for i in range(n_upsample_blocks+1):
            if i == 0:
                x = down_block(encoder_filters[n_upsample_blocks-i],
                               i, 0, use_batchnorm=use_batchnorm) (input)
            else:
                down_rate = to_tuple(upsample_rates[n_upsample_blocks-i-1])
                
                x = MaxPool2D(pool_size=down_rate, name='encoder_stage{}-0_maxpool'.format(i)) (x)
                x = down_block(encoder_filters[n_upsample_blocks-i],
                               i, 0, use_batchnorm=use_batchnorm) (x)
            downterm[i] = x

        x = downterm[-1]
        skip_layers_list = [downterm[n_upsample_blocks-i-1] for i in range(len(downterm[:-1]))]

    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        upsample_rate = to_tuple(upsample_rates[i])

        if i < len(skip_layers_list):
            if use_backbone:
                skip_connection = backbone.layers[skip_layers_list[i]].output
            else:
                skip_connection = skip_layers_list[i]
                
            if attention:
                skip_connection = attention_block(decoder_filters[i], skip_connection, n_upsample_blocks-i-1,
                                                  i+1, upsample_rate=upsample_rate) (x)

        x = up_block(decoder_filters[i], n_upsample_blocks-i-1, i+1, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm) (x)
        
        
    if use_backbone and 'vgg' not in backbone.name:
        x = up_block(decoder_filters[-1], 0, n_upsample_blocks+1, 
                     upsample_rate=to_tuple(upsample_rates[-1]),
                     skip=None, use_batchnorm=use_batchnorm) (x)

    x = Conv2D(classes, (3,3), padding='same', name='final_conv') (x)
    x = Activation(activation, name=activation) (x)

    return Model(input, x)