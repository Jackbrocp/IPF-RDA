# IPF-RDA: An Information-Preserving Framework for Robust Data Augmentation

This is the official implementation of IPF-RDA (http://arxiv.org/abs/), as was used for the paper.
 
You can directly start off using our implementations on CIFAR-10 and CIFAR-100.
## Use IPF-RDA for data augmentation
- Clone this directory and `cd`  into it.
 
`git clone https://github.com/Jackbrocp/IPF-RDA` 

`cd IPF-RDA`

## Updates
- 2023//: Initial release

## Getting Started
### Requirements
- Python 3
- PyTorch 1.6.0
- Torchvision 0.7.0
- Numpy
<!-- Install a fitting Pytorch version for your setup with GPU support, as our implementation  -->

### Train Examples 
#### Download the CIFAR dataset
Put the downloaded datasets into  ```./data/CIFAR10(or CIFAR100)/```

#### Download the results of CDIEA on CIFAR-10/100
[CIFAR-10](https://drive.google.com/file/d/18Kx7m7RkW4GtQjK-ltWbrJxKltg-aQIp/view?usp=sharing)

[CIFAR-100](https://drive.google.com/file/d/12mkyXfyrzmcSBBg4Cca5H6u6mbrhnzmC/view?usp=sharing)

Download the results and put them into  ```./data/CIFAR10(or CIFAR100)/```.
#### Parameters
```--conf```，path to the config file, e.g., ```confs/resnet18.yaml```
#### Examples 
Integrate Cutout into IPF-RDA as a robust data augmentation method to train the ResNet-18 model on CIFAR-10/100 datasets. 

```python train.py --conf confs/resnet18.yaml --aug 'cutout' --dataset 'CIFAR10' --cutout_length 16```

```python train.py --conf confs/resnet18.yaml --aug 'cutout' --dataset 'CIFAR100' --cutout_length 8```

#### More Examples
Integrate AutoAugment into IPF-RDA as a robust data augmentation method to train the ResNet-18 model on CIFAR-10/100 datasets. 

```python train.py --conf confs/resnet18.yaml --aug 'autoaugment' --dataset 'CIFAR10' --cutout_length 16 --fast_level 2 ```

```python train.py --conf confs/resnet18.yaml --aug 'autoaugment' --dataset 'CIFAR100' --cutout_length 8 --fast_level 2 ```

## Citation
If you find this repository useful in your research, please cite our paper:

`
citation
`
