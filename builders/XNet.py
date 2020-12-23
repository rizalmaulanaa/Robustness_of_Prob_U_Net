# https://github.com/MrGiovanni/UNetPlusPlus/blob/master/keras/segmentation_models/xnet/builder.py

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Average

from utils import get_layer_number, to_tuple
from blocks import Transpose2D_block, Upsample2D_block
from blocks import down_block, attention_block, DeepSupervision

def build_xnet(use_backbone, backbone, classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               input_shape=(None,None,3),
               use_batchnorm=True,
               attention=False,
               deep_supervision=False):

    downterm = [None] * (n_upsample_blocks+1)
    interm = [None] * (n_upsample_blocks+1) * (n_upsample_blocks+1)

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # Using Backbone for the Encoder
    if use_backbone:
        input = backbone.input
        output = backbone.output

        if len(skip_connection_layers) > n_upsample_blocks:
            downsampling_layers = skip_connection_layers[int(len(skip_connection_layers)/2):]
            skip_connection_layers = skip_connection_layers[:int(len(skip_connection_layers)/2)]
        else:
            downsampling_layers = skip_connection_layers

        # convert layer names to indices
        skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                                for l in skip_connection_layers])
        skip_layers_list = [backbone.layers[skip_connection_idx[i]].output
                            for i in range(len(skip_connection_idx))]
        downsampling_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                                   for l in downsampling_layers])
        downsampling_list = [backbone.layers[downsampling_idx[i]].output
                             for i in range(len(downsampling_idx))]

        for i in range(len(downsampling_idx)):
            if downsampling_list[0].name == backbone.output.name:
                # print("VGG16 should be!")
                downterm[n_upsample_blocks-i] = downsampling_list[i]
            else:
                downterm[n_upsample_blocks-i-1] = downsampling_list[i]

        downterm[-1] = output

    # Using Conv+relu for the Encoder
    else:
        encoder_filters = (decoder_filters[0]*2,) + decoder_filters
        input = Input(shape=input_shape)

        for i in range(n_upsample_blocks+1):
            if i==0:
                x = down_block(encoder_filters[n_upsample_blocks-i],
                               i, 0, use_batchnorm=use_batchnorm) (input)
            else:
                down_rate = to_tuple(upsample_rates[n_upsample_blocks-i-1])
                x = MaxPool2D(pool_size=down_rate) (x)
                x = down_block(encoder_filters[n_upsample_blocks-i],
                               i, 0, use_batchnorm=use_batchnorm) (x)
            downterm[i] = x

        skip_layers_list = [downterm[n_upsample_blocks-i-1]
                            for i in range(len(downterm[:-1]))]

        skip_connection_layers = tuple([i.name for i in skip_layers_list])
        output = downterm[-1]

    for i in range(len(skip_connection_layers)):
        interm[-i*(n_upsample_blocks+1)+(n_upsample_blocks+1)*(n_upsample_blocks-1)] = skip_layers_list[i]
    interm[(n_upsample_blocks+1)*n_upsample_blocks] = output

    for j in range(n_upsample_blocks):
        for i in range(n_upsample_blocks-j):
            upsample_rate = to_tuple(upsample_rates[i])

            if i == 0 and j < n_upsample_blocks-1 and len(skip_connection_layers) < n_upsample_blocks:
                interm[(n_upsample_blocks+1)*i+j+1] = None

            elif j == 0:
                if downterm[i+1] is not None:
                    if attention:
                        interm[(n_upsample_blocks+1)*i+j] = attention_block(decoder_filters[n_upsample_blocks-i-1],
                                                                            interm[(n_upsample_blocks+1)*i+j],
                                                                            i, j) ((downterm[i+1]))

                    interm[(n_upsample_blocks+1)*i+j+1] = up_block(decoder_filters[n_upsample_blocks-i-1],
                                                                   i, j+1, upsample_rate=upsample_rate,
                                                                   skip=interm[(n_upsample_blocks+1)*i+j],
                                                                   use_batchnorm=use_batchnorm)(downterm[i+1])
                else:
                    interm[(n_upsample_blocks+1)*i+j+1] = None

            else:
                if attention:
                    interm[(n_upsample_blocks+1)*i+j] = attention_block(decoder_filters[n_upsample_blocks-i-1],
                                                                        interm[(n_upsample_blocks+1)*i+j],
                                                                        i, j) (interm[(n_upsample_blocks+1)*(i+1)+j])

                interm[(n_upsample_blocks+1)*i+j+1] = up_block(decoder_filters[n_upsample_blocks-i-1],
                                                               i, j+1, upsample_rate=upsample_rate,
                                                               skip=interm[(n_upsample_blocks+1)*i : (n_upsample_blocks+1)*i+j+1],
                                                               use_batchnorm=use_batchnorm)(interm[(n_upsample_blocks+1)*(i+1)+j])

    # Deep Supervision
    if deep_supervision and (len(skip_connection_layers) == n_upsample_blocks):
        # Currently only VGG or not using backbone
        x = DeepSupervision(classes) (interm[1:n_upsample_blocks+1])
        x = Average(name='average_ds') (x)
    else:
        x = Conv2D(classes, (3,3), padding='same', name='final_conv')(interm[n_upsample_blocks])
        x = Activation(activation, name=activation)(x)

    return Model(input, x)
