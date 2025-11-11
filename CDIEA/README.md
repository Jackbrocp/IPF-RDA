# IPF-RDA-CDIEA
The implementation of CDIEA, as was used for the paper.

You can directly start off using our implemtantion on CIFAR-10/100.

## Command line parameters
```-g```, ```--gpu```, the GPU used, the default is 0

```--model_name```, the model used, the default is resnet18

```--data_name```, the dataset used, the default is cifar10

```--batch_size```, batch_size, the default is 10

```--attacker```, the attack method, the default is ours

## Usage Example
```python main_attack.py --g 0 --model_name resnet18 --data_name cifar10 --batch_size 1 --attacker ours```

## Result
You can find it in the ```./result``` folder.