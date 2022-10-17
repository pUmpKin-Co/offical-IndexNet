# IndexNet

![Fig8](https://pumpkintypora.oss-cn-shanghai.aliyuncs.com/imgFig8.png)

This is the official implementation of the IndexNet.

## Code

+ Code for building model: `IndexNetModel`
+ Code for data dataset、augmentation: `dataset`
+ Implementation of Deeplabv3+ segmentation model:`deeplabv3plus`
+ Other utils: `utils`
+ PyTorch-Lightning Model: `trainer.py`
+ Config: `config.py`

### Reference

Many Thanks to: [solo-learn](https://github.com/vturrisi/solo-learn)、[DetCon](https://github.com/deepmind/detcon)、[Pytorch-BYOL](https://github.com/sthalles/PyTorch-BYOL)

## Pre-trained model

### Potsdam

| Pre-train Dataset | Architecture | Epoch |                             Link                             |
| :---------------: | :----------: | :---: | :----------------------------------------------------------: |
|      Potsdam      |  ResNet-50   |  100  | [BaiDu](https://pan.baidu.com/s/11YXdmbcDT3dVtciM2ePfFg) (Code: tick) |
|      Potsdam      |  ResNet-50   |  400  | [BaiDu](https://pan.baidu.com/s/1PpC7rtq7QmxhDNHGSA13VQ) (Code: ebfk) |

### LoveDA

| Pre-train Dataset | Architecture | Epoch |                             Link                             |
| :---------------: | :----------: | :---: | :----------------------------------------------------------: |
|  LoveDA (Urban)   |  ResNet-50   |  100  | [BaiDu](https://pan.baidu.com/s/1DoCyklY3QAhK8X5SAVgxJw) (Code:m3qf) |
|  LoveDA (Urban)   |  ResNet-50   |  400  | [BaiDu](https://pan.baidu.com/s/1UfqTPfsUsb_s_9ZIeLHjLQ) (Code:4pmg) |

### SeCo

| Pre-train Dataset | Architecture | Epoch |                             Link                             |
| :---------------: | :----------: | :---: | :----------------------------------------------------------: |
|     SeCo-100K     |  ResNet-50   |  200  | [BaiDu](https://pan.baidu.com/s/1i_gEjtZI2VKAgGyIJdX9ag) (Code:g997) |

