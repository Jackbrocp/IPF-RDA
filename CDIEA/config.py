import torch
import random
import numpy as np
import os
import time


def mkdirs(dir_list):
    for dir_path in dir_list:
        if not os.path.exists(dir_path):
            print("mkdir!", dir_path)
            os.makedirs(dir_path)


class Config():
    def __init__(self, gpu=None, device=None, seed=10, target="0", model_name="resnet", data_name="imagenet", attacker="ours", is_pruning="0"):
        self.device = device
        self.seed = seed
        self.gpu = gpu
        self.process_id = None
        self.is_pruning = is_pruning
        self.target = target
        self.is_group = "0"

        self.mask_c = 1   

        self.sparsity = 0    

        self.attacker = attacker
        self.save_id = 0   

        self.pixel_max = 1
        self.pixel_min = 0

        self.model_name = model_name
        if model_name == "resnet18":
            self.model_path = "./save/resnet18.pth"

        self.data_name = data_name

        time_stamp = str(time.time())
        self.result_dir = os.path.join(
            "./result/", data_name, attacker, time_stamp, "adv_image")
        self.process_result_dir = os.path.join(
            "./result/", data_name, attacker, time_stamp, "process_result")
        self.pruner_result_dir = os.path.join(
            "./result/", data_name, attacker, time_stamp, "pruner_result")
        self.loss_dir = os.path.join(
            "./result/", data_name, attacker, time_stamp, "loss")
        self.compress_list_file = os.path.join(
            "./result/", data_name, attacker, time_stamp, "pruner_result.txt")
        mkdirs([self.result_dir, self.process_result_dir,
               self.pruner_result_dir, self.loss_dir])

        self.decay_factor = 0.9

        if data_name in ["cifar10", "cifar100"]:
            self.image_c = 3
            self.image_h = 32
            self.image_w = 32
            self.image_size = self.image_h * self.image_w
            self.num_classes = 10 if data_name == "cifar10" else 100
            self.eps = 4 * (self.pixel_max - self.pixel_min) / 255

            self.eps_l2 = 1.2
            self.data_dir = "./data/cifar10/" if data_name == "cifar10" else "./data/cifar100/"
            # normalize
            self.mean = torch.Tensor([0.4914, 0.4822, 0.4465]).view(
                1, 3, 1, 1).to(device)
            self.std = torch.Tensor([0.2023, 0.1994, 0.2010]).view(
                1, 3, 1, 1).to(device)
            self.normalized_ops = (self.mean, self.std)
            self.max_init_count = 1
            self.pruner = "fc"
            self.C = 1      
            if target == "0":
                self.alpha_s = 0.1
                self.alpha_e = 100
                self.epochs = 6
                self.iterations = 60
                self.loss_lambda_gamma = 10.0
                self.C = 1
            elif target == "1":
                self.alpha_s = 0.1
                self.alpha_e = 100
                if self.pruner == "fcn":
                    self.alpha_e = 10
                if self.pruner == "unet":
                    self.alpha_e = 20
                self.epochs = 8
                self.iterations = 80
                self.loss_lambda_gamma = 10.0
                if attacker == "cw_l2":
                    self.epochs = 10
                    self.iterations = 100
        self.mask_h = self.image_h
        self.mask_w = self.image_w
        self.mask_size = self.mask_h * self.mask_w


    def setup_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
