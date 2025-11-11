import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

from model.BasicCNN import BasicCNN
from model.VGG import VGG
from model.resnet18 import ResNet18

# 设置模型的requires_grad属性
def set_parameter_requires_grad(model, feature_extracting = True):
    # 默认情况下，加载的预训练模型的所有参数的requires_grad都是True，这里需要设置为False
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# 根据输出层的个数，重塑模型结构
def initialize_model(model_name, num_classes, model_path, gpus, feature_extract = True, use_pretrained=True):
    device = torch.device("cuda:" + gpus if torch.cuda.is_available() else "cpu")
    if model_name in ["inception_v3", "mobilenet_v2"]:
        if model_name == "inception_v3":
            model = models.inception_v3(pretrained=use_pretrained)  # inception_v3，使用pytorch自带的，测试发现精度为77.212%
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=use_pretrained)      # 用于测量迁移性的
        elif model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=use_pretrained)  # 用于测量迁移性的
        set_parameter_requires_grad(model, feature_extract)
        if torch.cuda.device_count() > 1:
            device_ids = [int(x) for x in gpus.split(",")]
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)
    else:
        # 自己训练的模型
        if model_name == "vgg19":
            model = VGG("VGG19", num_classes)
        elif model_name == "vgg13":
            model = VGG("VGG13", num_classes)
        elif model_name == "basicCNN":
            model = BasicCNN()
        elif model_name in ["resnet18", "resnet18_adv"]:
            model = ResNet18(num_classes)
        elif model_name in ["resnet50", "flower_resnet50", "flower_resnet50_baseaug"]:
            model = models.resnet50(pretrained=True)
            model.avgpool = nn.AdaptiveAvgPool2d(1)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)

        if torch.cuda.device_count() > 1:
            device_ids = [int(x) for x in gpus.split(",")]
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        
        model = model.to(device)
        model_dict = torch.load(model_path) if model_name == "resnet50" else torch.load(model_path, map_location=torch.device("cpu")).module.state_dict()

        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(model_dict)
        else:
            model.load_state_dict(model_dict)
        
        '''
        model = model.to(device)
        model_dict = torch.load(model_path, map_location=torch.device("cpu")).module.state_dict()
        model.module.load_state_dict(model_dict)
        '''

    return model
