# modified code from https://github.com/MrGiovanni/UNetPlusPlus/blob/master/keras/segmentation_models

from builders.UNet import build_unet
from builders.XNet import build_xnet
from builders.AttentionXNet import build_Attxnet
from utils import freeze_model
from backbones import get_backbone

DEFAULT_SKIP_CONNECTIONS = {
    'vgg16':            ('block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
    'vgg19':            ('block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'),
    'resnet50':         ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu'),
    'resnet101':        ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu'),
    'resnet152':        ('conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu'),
    'densenet121':          (311, 139, 51, 4),
    'densenet169':          (367, 139, 51, 4),
    'densenet201':          (479, 139, 51, 4),
}

def AttXnet(use_backbone,
            backbone_name='vgg16',
            input_shape=(None, None, 3),
            input_tensor=None,
            encoder_weights='imagenet',
            freeze_encoder=False,
            skip_connections='default',
            decoder_block_type='upsampling',
            decoder_filters=(256,128,64,32,16),
            decoder_use_batchnorm=True,
            n_upsample_blocks=4,
            upsample_rates=(2,2,2,2),
            classes=1,
            activation='sigmoid',
            attention=True,
            deep_supervision=False):
    """
    Args:
        use_backbone: (bool) if True then using backbone from Keras, if False then using downsample block
        backbone_name: (str) look at list of available backbones.
        input_shape:  (tuple) dimensions of input data (H, W, C)
        input_tensor: keras tensor
        encoder_weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet)
        freeze_encoder: (bool) Set encoder layers weights as non-trainable. Useful for fine-tuning
        skip_connections: if 'default' is used take default skip connections,
            else provide a list of layer numbers or names starting from top of model
        decoder_block_type: (str) one of 'upsampling' and 'transpose' (look at blocks.py)
        decoder_filters: (int) number of convolution layer filters in decoder blocks
        decoder_use_batchnorm: (bool) if True add batch normalisation layer between `Conv2D` ad `Activation` layers
        n_upsample_blocks: (int) a number of upsampling blocks
        upsample_rates: (tuple of int) upsampling rates decoder blocks
        classes: (int) a number of classes for output
        activation: (str) one of keras activations for last model layer
        attention: (bool) if True then used attention block, else then not used attention block
        deep_supervision: (bool) if True then used deep supervision on segmentation branch, else then not used deep supervision
    Returns:
        keras.models.Model instance
    """

    att_name = 'Att' if attention else ''
    ds_name = '_ds' if deep_supervision else ''
    if use_backbone:
        backbone = get_backbone(backbone_name,
                                input_shape=input_shape,
                                input_tensor=input_tensor,
                                weights=encoder_weights,
                                include_top=False)

        if skip_connections == 'default':
            skip_connections = DEFAULT_SKIP_CONNECTIONS[backbone_name][-(n_upsample_blocks):]

        model_name = '{}X{}{}'.format(att_name, '_'+backbone_name, ds_name)
    else:
        backbone = None
        skip_connections = None
        model_name = '{}X{}{}'.format(att_name, '_enc', ds_name)

    model = build_Attxnet(use_backbone,
                       backbone,
                       classes,
                       skip_connections,
                       decoder_filters=decoder_filters,
                       block_type=decoder_block_type,
                       activation=activation,
                       n_upsample_blocks=n_upsample_blocks,
                       upsample_rates=upsample_rates,
                       use_batchnorm=decoder_use_batchnorm,
                       input_shape=input_shape,
                       attention=attention,
                       deep_supervision=deep_supervision)

    # lock encoder weights on backbone for fine-tuning
    if freeze_encoder:
        freeze_model(backbone)

    model._name = model_name

    return model

def Xnet(use_backbone,
         backbone_name='vgg16',
         input_shape=(None, None, 3),
         input_tensor=None,
         encoder_weights='imagenet',
         freeze_encoder=False,
         skip_connections='default',
         decoder_block_type='upsampling',
         decoder_filters=(256,128,64,32,16),
         decoder_use_batchnorm=True,
         n_upsample_blocks=4,
         upsample_rates=(2,2,2,2),
         classes=1,
         activation='sigmoid',
         attention=False,
         deep_supervision=False):
    """
    Args:
        use_backbone: (bool) if True then using backbone from Keras, if False then using downsample block
        backbone_name: (str) look at list of available backbones.
        input_shape:  (tuple) dimensions of input data (H, W, C)
        input_tensor: keras tensor
        encoder_weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet)
        freeze_encoder: (bool) Set encoder layers weights as non-trainable. Useful for fine-tuning
        skip_connections: if 'default' is used take default skip connections,
            else provide a list of layer numbers or names starting from top of model
        decoder_block_type: (str) one of 'upsampling' and 'transpose' (look at blocks.py)
        decoder_filters: (int) number of convolution layer filters in decoder blocks
        decoder_use_batchnorm: (bool) if True add batch normalisation layer between `Conv2D` ad `Activation` layers
        n_upsample_blocks: (int) a number of upsampling blocks
        upsample_rates: (tuple of int) upsampling rates decoder blocks
        classes: (int) a number of classes for output
        activation: (str) one of keras activations for last model layer
        attention: (bool) if True then used attention block, else then not used attention block
        deep_supervision: (bool) if True then used deep supervision on segmentation branch, else then not used deep supervision
    Returns:
        keras.models.Model instance
    """

    att_name = 'Att' if attention else ''
    ds_name = '_ds' if deep_supervision else ''
    if use_backbone:
        backbone = get_backbone(backbone_name,
                                input_shape=input_shape,
                                input_tensor=input_tensor,
                                weights=encoder_weights,
                                include_top=False)

        if skip_connections == 'default':
            skip_connections = DEFAULT_SKIP_CONNECTIONS[backbone_name][-(n_upsample_blocks):]

        model_name = '{}X{}{}'.format(att_name, '_'+backbone_name, ds_name)
    else:
        backbone = None
        skip_connections = None
        model_name = '{}X{}{}'.format(att_name, '_enc', ds_name)

    model = build_xnet(use_backbone,
                       backbone,
                       classes,
                       skip_connections,
                       decoder_filters=decoder_filters,
                       block_type=decoder_block_type,
                       activation=activation,
                       n_upsample_blocks=n_upsample_blocks,
                       upsample_rates=upsample_rates,
                       use_batchnorm=decoder_use_batchnorm,
                       input_shape=input_shape,
                       attention=attention,
                       deep_supervision=deep_supervision)

    # lock encoder weights on backbone for fine-tuning
    if freeze_encoder:
        freeze_model(backbone)

    model._name = model_name

    return model

def Unet(use_backbone,
         backbone_name='vgg16',
         input_shape=(None, None, 3),
         input_tensor=None,
         encoder_weights='imagenet',
         freeze_encoder=False,
         skip_connections='default',
         decoder_block_type='upsampling',
         decoder_filters=(256,128,64,32,16),
         decoder_use_batchnorm=True,
         n_upsample_blocks=4,
         upsample_rates=(2,2,2,2),
         classes=1,
         activation='sigmoid',
         attention=False):
    """
    Args:
        use_backbone: (bool) if True then using backbone from Keras, if False then using downsample block
        backbone_name: (str) look at list of available backbones.
        input_shape:  (tuple) dimensions of input data (H, W, C)
        input_tensor: keras tensor
        encoder_weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet)
        freeze_encoder: (bool) Set encoder layers weights as non-trainable. Useful for fine-tuning
        skip_connections: if 'default' is used take default skip connections,
            else provide a list of layer numbers or names starting from top of model
        decoder_block_type: (str) one of 'upsampling' and 'transpose' (look at blocks.py)
        decoder_filters: (int) number of convolution layer filters in decoder blocks
        decoder_use_batchnorm: (bool) if True add batch normalisation layer between `Conv2D` ad `Activation` layers
        n_upsample_blocks: (int) a number of upsampling blocks
        upsample_rates: (tuple of int) upsampling rates decoder blocks
        classes: (int) a number of classes for output
        activation: (str) one of keras activations for last model layer
        attention: (bool) if True then used attention block, else then not used attention block
    Returns:
        keras.models.Model instance
    """

    att_name = 'Att' if attention else ''
    if use_backbone:
        backbone = get_backbone(backbone_name,
                                input_shape=input_shape,
                                input_tensor=input_tensor,
                                weights=encoder_weights,
                                include_top=False)

        if skip_connections == 'default':
            skip_connections = DEFAULT_SKIP_CONNECTIONS[backbone_name][-(n_upsample_blocks):]

        model_name = '{}U{}'.format(att_name, '_'+backbone_name)
    else:
        backbone = None
        skip_connections = None
        model_name = '{}U{}'.format(att_name, '_enc')

    model = build_unet(use_backbone,
                       backbone,
                       classes,
                       skip_connections,
                       decoder_filters=decoder_filters,
                       block_type=decoder_block_type,
                       activation=activation,
                       n_upsample_blocks=n_upsample_blocks,
                       upsample_rates=upsample_rates,
                       input_shape=input_shape,
                       use_batchnorm=decoder_use_batchnorm,
                       attention=attention)

    # lock encoder weights on backbone for fine-tuning
    if freeze_encoder:
        freeze_model(backbone)

    model._name = model_name

    return model
