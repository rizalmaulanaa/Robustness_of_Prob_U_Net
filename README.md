# Robustness of Probabilistic U-Net
White Matter Hyperintensities (WMHs) segmentation using a modified U-Net architecture. Mainly this repository got inspiration and insight on repositories [UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus) and [probabilistic_unet](https://github.com/SimonKohl/probabilistic_unet). In this repository, you can use many kinds of modified U-Net architecture, such as U-Net, Attention U-Net, U-Net++, Attention U-Net++, and Probabilistic U-Net. The implementation is using TensorFlow and Keras, please check the [requirements.txt](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/requirements.txt) for the version that I used.

You can access the pre-trained models with these link: https://drive.google.com/drive/folders/1-G8h1XcnFNcFzAg2OdsxVa-3kF-sM6RA?usp=sharing

## U-Net
Title: U-Net: Convolutional Networks for Biomedical Image Segmentation.<br>
Authors: Olaf Ronneberger, Philipp Fischer, and Thomas Brox.<br>
[[paper](https://arxiv.org/abs/1505.04597)]<br>
The architecture of U-Net:<br>
![UNet](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/model%20img/Models-U_Net.png)<br>

## Attention U-Net
Title: Attention U-Net: Learning Where to Look for the Pancreas.<br>
Authors: Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa, Kensaku Mori, Steven McDonagh, Nils Y Hammerla, Bernhard Kainz, Ben Glocker, Daniel Rueckert.<br>
[[paper](https://arxiv.org/abs/1804.03999)]<br>
Attention Gate:<br>
![Attention Gate](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/model%20img/Models-Attention_Gate.png)<br>
The architecture of Attention U-Net:<br>
![Attention UNet](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/model%20img/Models-Attention_U_Net.png)<br>

## U-Net++
Title: UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation.<br>
Authors: Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang.<br>
[[paper](https://arxiv.org/abs/1912.05074)]<br>
The architecture of U-Net++:<br>
![UNet++](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/model%20img/Models-U_Net%2B%2B.png)<br>

## Attention U-Net++
Title: Attention Unet++: A Nested Attention-Aware U-Net for Liver CT Image Segmentation.<br>
Authors: Chen Li, Yusong Tan, Wei Chen, Xin Luo, Yuanming Gao, Xiaogang Jia, Zhiying Wang.<br>
[[paper](https://ieeexplore.ieee.org/document/9190761)]<br>
The architecture of Attention U-Net++:<br>
![Attention UNet++](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/model%20img/Models-Attention_U_Net%2B%2B.png)<br>

## Probabilistic U-Net
Title: A Probabilistic U-Net for Segmentation of Ambiguous Images.<br>
Authors: Simon A. A. Kohl, Bernardino Romera-Paredes, Clemens Meyer, Jeffrey De Fauw, Joseph R. Ledsam, Klaus H. Maier-Hein, S. M. Ali Eslami, Danilo Jimenez Rezende, Olaf Ronneberger.<br>
[[paper](https://arxiv.org/abs/1806.05034)]<br>
The architecture of Probabilistic U-Net:<br>
Training process:<br>
![Training](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/model%20img/Models-Probabilistic_U_Net-training.png)<br>
Sampling process:<br>
![Sampling](https://github.com/rizalmaulanaa/Attention-XNet/blob/master/model%20img/Models-Probabilistic_U_Net-sampling.png)<br>

## Usage
To use this repository, simply just call the (model name) function in model file.
```python
import models.models as model
model = model.Unet(use_backbone=False, input_shape=(256,256,1), attention=False)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(img_train, seg_train, epochs=100, batch_size=32)
```
For the deterministic models, you can choose to use backbone or only (conv+relu)x2. If you want to use backbone please fill the parameter use_backbone into True. The backbone that compatible with this implementation is VGG, ResNet, and DenseNet. And if you want to use Attention Gate on modified U-Net architecture, please fill the parameter attention into True.
