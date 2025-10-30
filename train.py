import pathlib
import sys
import os
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument('--log_interval',type=int,default=50,help='log training status')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--conf', default='./confs/resnet18.yaml', type=str,  help=' yaml file')
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--gpus',type=str,default='6,7')
parser.add_argument('--resume',type=str,default=None)
parser.add_argument('--aug_type',type=str, default='randaugment')
parser.add_argument('--cutout_length',type=int, default=16)
parser.add_argument('--dataset',type=str, required=True)
parser.add_argument('--save_model',type=bool, default=False)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpus
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch import optim
import random
from Dataset import CIFAR10Dataset, CIFAR100Dataset
from Network import *
from IPF import  ipf_data_aug, Grid, HaS, trivialaugment
import yaml
from warmup_scheduler import GradualWarmupScheduler
import copy
cuda = True if torch.cuda.is_available() else False

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(args.seed)
with open(args.conf) as f:
    cfg = yaml.safe_load(f)
if args.aug_type in ['cutout','randomerasing','cutmix']:
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32,padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])
    ori_transform = None

elif args.aug_type in ['autoaugment','randaugment','fastaugment','trivialaugment','has','gridmask']:
    ori_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32,padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        ])
    transform = copy.deepcopy(ori_transform)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])
 
print('============Augmentation==============:',  args.aug_type)
 
best_acc = 0
best_epoch = 0
acc_list = []
momentum = args.momentum
model = get_model(cfg['model']['type'],num_classes=num_class(args.dataset.lower()))
model = torch.nn.DataParallel(model, device_ids=np.arange(len(args.gpus.split(','))).tolist()).cuda()
if cfg['optimizer']['type'] == 'sgd':
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg['lr'],
        momentum=momentum,
        weight_decay=cfg['optimizer']['decay'],
        nesterov=cfg['optimizer']['nesterov']
    )
lr_schduler_type = cfg['lr_schedule']['type']
if lr_schduler_type == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epoch'], eta_min=0.)
elif lr_schduler_type == 'step':
    scheduler =  torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['lr_schedule']['milestones'],gamma=cfg['lr_schedule']['gamma'])
if cfg['lr_schedule']['warmup']!='' and  cfg['lr_schedule']['warmup']['epoch'] > 0:
    scheduler =  GradualWarmupScheduler(
        optimizer,
        multiplier = cfg['lr_schedule']['warmup']['multiplier'],
        total_epoch = cfg['lr_schedule']['warmup']['epoch'],
        after_scheduler = scheduler
    )
epoches = cfg['epoch']
batch = cfg['batch']
loss = nn.CrossEntropyLoss()
def load_threshold_list():
    print('====================load threshold====================')
    path = './data/{}/importance_threshold_length_{}_ratio_60.pt'.format(args.dataset.upper(),args.cutout_length)
    return torch.load(path)
def load_mask_list():
    print('====================load mask====================')
    print('./data/{}/mask.pt'.format(args.dataset.upper()))
    path = './data/{}/mask.pt'.format(args.dataset.upper())
    return torch.load(path)
def load_candidate():
    if args.dataset == 'CIFAR10':
        # path = './data/{}/CIFAR10_candidate_box_list.pt'.format(args.dataset)
        # return np.array(torch.load(path))
        path = '/nas/yangsuorong/KeepAugment-Pro/data/{}/box_list_sum_geq.pt'.format(args.dataset)
        return np.array(torch.load(path))
    else:
        path = './data/{}/CIFAR100_candidate_box_list.npy'.format(args.dataset)
        return np.load(path, allow_pickle=True)
    
ipf_type = 2

if args.aug_type == 'cutout':
    ipf_rda = ipf_data_aug.Cutout_IPF(length=args.cutout_length)
    ipf_type = 1
elif args.aug_type == 'gridmask':
    ipf_rda = ipf_data_aug.GridMask_IPF(args.cutout_length)
    transform.transforms.append(Grid(d1=24,d2=33, rotate=1, ratio=0.4,mode=1, prob=1.))
elif args.aug_type == 'has':
    ipf_rda = ipf_data_aug.HaS_IPF(args.cutout_length)
    transform.transforms.append(HaS())
elif args.aug_type == 'cutmix':
    ipf_rda = ipf_data_aug.CutMix_IPF(length=args.cutout_length)
    ipf_type = 1
elif args.aug_type == 'randomerasing':
    ipf_rda = ipf_data_aug.RandomErasing_IPF(length=args.cutout_length)
    ipf_type = 1
elif args.aug_type == 'autoaugment':
    ipf_rda = ipf_data_aug.Autoaugment_IPF(length=args.cutout_length)
    transform.transforms.insert(1, ipf_data_aug.CIFAR10Policy())
elif args.aug_type == 'randaugment':
    ipf_rda = ipf_data_aug.Randaugment_IPF(length=args.cutout_length, N=1, M=30)
    transform.transforms.insert(1, ipf_data_aug.RandAugment(1,30))
elif args.aug_type == 'fastaugment':
    ipf_rda = ipf_data_aug.FastAugmentation_IPF(length=args.cutout_length)
    transform.transforms.insert(1, ipf_data_aug.FastAugment(ipf_data_aug.fa_reduced_cifar10()))
elif args.aug_type == 'trivialaugment':
    ipf_rda = ipf_data_aug.Trivialaugment_IPF(length=args.cutout_length)
    trivialaugment.set_augmentation_space(augmentation_space='standard',num_strengths=30)
    TAugment = trivialaugment.TrivialAugment()
    transform.transforms.insert(1, TAugment)

if ipf_type == 1:
    threshold_list = load_threshold_list().cuda()
    mask_list = load_mask_list().cuda()
elif ipf_type == 2 :
    print('====================load candidate====================')
    candidate_list = load_candidate()
else:
    threshold_list = load_threshold_list().cuda()
    mask_list = load_mask_list().cuda()
    candidate_list = load_candidate()
    

if args.dataset == 'CIFAR10':
    root = 'data/CIFAR10/'
    trainset = CIFAR10Dataset(root=root,
                        train=True,
                        ori_transform= ori_transform,
                        transform=transform)
    testset = CIFAR10Dataset(root=root,
                        train=False,
                        transform=transform_test)
elif args.dataset == 'CIFAR100':
    root = 'data/CIFAR100/'
    trainset = CIFAR100Dataset(root, train=True,fine_label=True, 
                               transform=transform,
                               ori_transform=ori_transform
                               )
    testset = CIFAR100Dataset(root, train=False,fine_label=True, 
                              transform=transform_test)



train_loader=DataLoader(dataset=trainset, batch_size=cfg['batch'],
                        shuffle=True,
                        num_workers=8,
                        pin_memory=True)
test_loader=DataLoader(dataset=testset, batch_size=cfg['batch'],
                        shuffle=False,
                        num_workers=8,
                        pin_memory=True)
start_epoch = 0
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}/{}'.format(args.dataset, args.resume))
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
def train(net, epoch):
    # global sample_ratio
    global optimizer
    global scheduler
    global loss
    net.train()
    training_loss=0.0
    total = len(train_loader.dataset)
    correct=0
    for i, data in enumerate(train_loader,0):
        if args.aug_type in ['autoaugment','randaugment','fastaugment','trivialaugment','gridmask','has'] :
            # selective paste
            ori_inputs, inputs, labels, filename = data
            ori_inputs, inputs, labels = ori_inputs.cuda(), inputs.cuda(), labels.cuda()
        else:
            # selective cut
            ori_inputs = None
            inputs, labels, filename = data
            inputs, labels = inputs.cuda(), labels.cuda()
        #############################
        file_list = [int(i[:-4]) for i in filename] 
        if ipf_type == 1:
            inputs = ipf_rda(inputs, mask_list[file_list], threshold_list[file_list])
        elif ipf_type == 2 :
            inputs = ipf_rda(ori_inputs, inputs, candidate_list[file_list])
        else:
            inputs = ipf_rda(ori_inputs, inputs, candidate_list[file_list], mask_list[file_list], threshold_list[file_list])
        #############################
        optimizer.zero_grad()
        outputs = net(inputs)
        l = loss(outputs, labels)
        l.backward()
        optimizer.step()
        training_loss+=l.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        if (i+1)% args.log_interval==0:
            loss_mean = training_loss/(i+1)
            trained_total  = (i+1)*len(labels)
            acc = 100. * correct/trained_total
            progress = 100. * trained_total/total
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}  Acc: {:.6f} '.format(epoch,
                trained_total, total, progress, loss_mean, acc ))
def test(net,epoch ):
    global best_acc
    global best_epoch
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs,targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = correct * 100. /total
    if acc>=best_acc:
        best_acc = acc
        best_epoch = epoch
    print('EPOCH:{}, ======================ACC:{}===================='.format(epoch, acc))
    print('BEST EPOCH:{},BEST ACC:{}'.format(best_epoch,best_acc))


if __name__ =='__main__':
    average_time = 0
    for epoch in tqdm(range(start_epoch ,epoches)):
        train(model, epoch  )
        test(model, epoch  )
        scheduler.step()
 
  