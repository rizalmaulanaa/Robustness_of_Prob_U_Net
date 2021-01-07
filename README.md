# Image Segmentation
Image Segmentation on Biomedical Task using a modified U-Net architecture. Mainly this repository got inspiration and insight on repository [UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus). In this repository, you can use many kinds of modified U-Net architecture, such as U-Net, Attention U-Net, UNet++, and Attention UNet++. The implementation is using TensorFlow and Keras, please check the [requirements.txt](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/requirements.txt) for the version that I used.

## UNet
Title: U-Net: Convolutional Networks for Biomedical Image Segmentation
Authors: Olaf Ronneberger, Philipp Fischer, and Thomas Brox
[paper](https://arxiv.org/abs/1505.04597)
The architecture of UNet:
![UNet]()

## Attention UNet
Title: Attention U-Net: Learning Where to Look for the Pancreas
Authors: Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa, Kensaku Mori, Steven McDonagh, Nils Y Hammerla, Bernhard Kainz, Ben Glocker, Daniel Rueckert
[paper](https://arxiv.org/abs/1804.03999)
Attention Gate:
![Attention Gate]()
The architecture of Attention UNet:
![Attention UNet]()

## UNet++
Title: UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation
Authors: Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang
[paper](https://arxiv.org/abs/1912.05074)
The architecture of UNet++:
![UNet++]()

## Attention UNet++
Title: Attention Unet++: A Nested Attention-Aware U-Net for Liver CT Image Segmentation
Authors: Chen Li, Yusong Tan, Wei Chen, Xin Luo, Yuanming Gao, Xiaogang Jia, Zhiying Wang
[paper](https://ieeexplore.ieee.org/document/9190761)
The architecture of Attention UNet++:
![Attention UNet++]()

## Usage
To use this repository, simply just call the (model name) function in model file.
```python
import model
model = model.Unet(use_backbone=False, input_shape=(256,256,1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coeff])
model.fit(img_train, seg_train, epochs=100, batch_size=32)
```
In this repository, you can choose to use backbone or only (conv+relu)x2. If you want to use backbone please fill the parameter use_backbone into True. The backbone that compatible with this implementation is VGG, ResNet, and DenseNet. And if you want to use Attention Gate on modified U-Net architecture, please fill the parameter attention into True.
