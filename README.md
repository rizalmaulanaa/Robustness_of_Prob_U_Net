# Image Segmentation
Image Segmentation on Biomedical Task using a modified U-Net architecture. Mainly this repository got inspiration and insight on repository [UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus). In this repository, you can use many kinds of modified U-Net architecture, such as U-Net, Attention U-Net, UNet++, and Attention UNet++. The implementation is using TensorFlow and Keras, please check the [requirements.txt](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/requirements.txt) for the version that I used.

## UNet
Title: U-Net: Convolutional Networks for Biomedical Image Segmentation.<br>
Authors: Olaf Ronneberger, Philipp Fischer, and Thomas Brox.<br>
[[paper](https://arxiv.org/abs/1505.04597)]<br>
The architecture of UNet:<br>
![UNet](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/model%20img/Models-U-Net.png)<br>

## Attention UNet
Title: Attention U-Net: Learning Where to Look for the Pancreas.<br>
Authors: Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa, Kensaku Mori, Steven McDonagh, Nils Y Hammerla, Bernhard Kainz, Ben Glocker, Daniel Rueckert.<br>
[[paper](https://arxiv.org/abs/1804.03999)]<br>
Attention Gate:<br>
![Attention Gate](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/model%20img/Models-Attention%20Gate.png)<br>
The architecture of Attention UNet:<br>
![Attention UNet](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/model%20img/Models-Attention%20U-Net.png)<br>

## UNet++
Title: UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation.<br>
Authors: Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang.<br>
[[paper](https://arxiv.org/abs/1912.05074)]<br>
The architecture of UNet++:<br>
![UNet++](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/model%20img/Models-UNet%2B%2B.png)<br>

## Attention UNet++
Title: Attention Unet++: A Nested Attention-Aware U-Net for Liver CT Image Segmentation.<br>
Authors: Chen Li, Yusong Tan, Wei Chen, Xin Luo, Yuanming Gao, Xiaogang Jia, Zhiying Wang.<br>
[[paper](https://ieeexplore.ieee.org/document/9190761)]<br>
The architecture of Attention UNet++:<br>
![Attention UNet++](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/model%20img/Models-Attention%20UNet%2B%2B%20with%20downsampling.png)<br>

## Usage
To use this repository, simply just call the (model name) function in model file.
```python
import model
model = model.Unet(use_backbone=False, input_shape=(256,256,1), attention=False)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(img_train, seg_train, epochs=100, batch_size=32)
```
In this repository, you can choose to use backbone or only (conv+relu)x2. If you want to use backbone please fill the parameter use_backbone into True. The backbone that compatible with this implementation is VGG, ResNet, and DenseNet. And if you want to use Attention Gate on modified U-Net architecture, please fill the parameter attention into True.
