import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def __init__(self, data_index, all_imgs, transform=None):
        self.all_imgs = all_imgs
        self.data_index = data_index
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.all_imgs[self.data_index[index]]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_index)


def normalization(images, mean_std):
    mean, std = mean_std
    result = (images - mean) / std
    return result


def inv_normalization(images, mean_std):
    mean, std = mean_std
    result = images * std + mean
    return result


def load_data(data_dir, batch_size, conf, data_name):
    if data_name in ["cifar10", "cifar100"]:
        # normalize = transforms.Normalize(mean=mean, std=std)
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ])
        '''
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=data_transforms)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader
        '''
        if data_name == "cifar10":
            train_dataset = datasets.CIFAR10(
                root=data_dir, train=True, transform=data_transforms, download=True)
        elif data_name == "cifar100":
            train_dataset = datasets.CIFAR100(
                root=data_dir, train=True, transform=data_transforms, download=True)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True)
        return train_loader

        '''
        if data_name == "cifar10":
            test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=data_transforms)
        elif data_name == "cifar100":
            test_dataset = datasets.CIFAR100(root=data_dir, train=False, transform=data_transforms)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True)
        return test_loader
        '''


def tensor_to_image(image, data_name):
    if data_name == "mnist":
        # inv_normalize = transforms.Normalize(mean=[-1], std=[2])  
        inv_transform = transforms.Compose([
            # inv_normalize,
            transforms.ToPILImage(),
        ])
        return inv_transform(image)  
    else:
        # inv_normalize = transforms.Normalize(mean=-(mean/std), std=1/std)  
        inv_transform = transforms.Compose([
            # inv_normalize,
            transforms.ToPILImage(),
        ])
        return inv_transform(image)


def save_pkl(data_list, name_list, result_dir, process_id=None):
    if process_id is None:
        for data, name in zip(data_list, name_list):
            if data is None:
                continue
            with open(os.path.join(result_dir, name + ".pkl"), "wb") as f:
                pickle.dump(data, f)
    else:
        for data, name in zip(data_list, name_list):
            if data is None:
                continue
            new_dir = os.path.join(result_dir, str(process_id))
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            with open(os.path.join(new_dir, name + ".pkl"), "wb") as f:
                pickle.dump(data, f)


def load_pkl(name_list, result_dir, process_id=None):
    data_list = []
    new_dir = result_dir if process_id is None else os.path.join(
        result_dir, str(process_id))
    for name in name_list:
        with open(os.path.join(new_dir, name + ".pkl"), 'rb') as f:
            data_list.append(pickle.load(f))
    return data_list



def save_train_process(change, index, loss_dir):
    # print(change)
    for name in ['losses', 'losses_class', 'losses_compress', 'compresses']:
        data_list = change[name][index]
        fig, ax = plt.subplots()
        plt.xlabel('iterations')
        plt.ylabel('loss')
        """set interval for y label"""
        yticks = np.arange(-0.1, 0.5, 0.01)
        """set min and max value for axes"""
        ax.set_ylim([-0.1, 0.5])
        x = range(1, len(data_list) + 1)
        plt.plot(x, data_list, label=name)
        """open the grid"""
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.savefig(os.path.join(loss_dir, str(index) + "_" + name + ".png"))
        plt.close("all")



def save_train_process_batch(change, index, loss_dir, save_id):
    # print(change)
    for name in ['losses', 'losses_class', 'losses_compress', 'compresses']:
        data_list = change[name][index]
        fig, ax = plt.subplots()
        plt.xlabel('iterations')
        plt.ylabel('loss')
        """set interval for y label"""
        yticks = np.arange(-0.1, 0.5, 0.01)
        """set min and max value for axes"""
        ax.set_ylim([-0.1, 0.5])
        x = range(1, len(data_list) + 1)
        plt.plot(x, data_list, label=name)
        """open the grid"""
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.savefig(os.path.join(loss_dir, str(save_id) + "_" + name + ".png"))
        plt.close("all")



def read_compress(filepath, default_compress_r=0.05):
    compress_list = []
    lines = []
    with open(filepath) as f:
        for line in f:
            lines.append(line)
    for index, line in enumerate(lines):
        if index == len(lines) - 1:
            break
        success = line.split(',')[0].split()[-1]
        if success == "1" or success == "True":
            compress = float(line.split("=")[-1].strip().split("%")[0]) * 0.01
        else:
            compress = default_compress_r   
        compress_list.append(compress)
    return compress_list



def get_target_labels(num_classes, y_true):
    y_target = np.random.randint(0, num_classes, size=y_true.shape[0])
    y_target = torch.LongTensor(y_target)
    for i in range(len(y_target)):
        if y_target[i] == y_true[i]:
            y_target[i] = (y_target[i] + 1) % num_classes
    return y_target



def get_mask(image, adv_image, data_name):
    mask = np.abs(image.numpy() - adv_image.numpy())
    mask = mask <= 1e-6 if data_name == "mnist" else np.all(
        mask <= 1e-6, axis=0)
    mask = 1 - mask
    mask = mask.reshape(mask.shape[0], mask.shape[1], 1).astype(np.uint8) * 255
    return mask
