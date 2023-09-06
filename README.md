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
#### Download the results of CDIEA on CIFAR-10/100
[CIFAR-10]()

[CIFAR-100]()

Download the results and put them into  ```.//```.
#### Parameters
```--conf```，path to the config file, e.g., ```confs/resnet18.yaml```
#### Examples 
Apply IPF-RDA as a data augmentation method to train the ResNet-18 model on CIFAR-10/100 datasets.

```python train.py --conf confs/resnet18.yaml```

#### More Examples
Run additional comparisons on AdvMask combined with other data augmentation methods. (e.g., "AdvMask+AutoAugment")
First, change ```mask``` parameter in the config file, e.g. "AutoAugment", "Fast-AutoAugment"

```python additional_comparison.py --conf confs/resnet18.yaml```

## Citation
If you find this repository useful in your research, please cite our paper:

`
citation
`
