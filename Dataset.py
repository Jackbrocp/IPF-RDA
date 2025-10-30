from torch.utils.data import Dataset
import os
import pickle
import numpy as np

class CIFAR10Dataset(Dataset):
    base_folder=''
    preix='adv_'
    train_list=[
        [preix+'data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        [preix+'data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        [preix+'data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        [preix+'data_batch_4', '634d18415352ddfa80567beed471001a'],
        [preix+'data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list=[
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    def __init__(self,root,train=True,ori_transform=None,transform=None, target_transform=None):
        super(CIFAR10Dataset,self).__init__()
        self.train=train
        self.root=root
        self.data: Any = []
        self.targets=[]
        self.filename_list=[]
        self.ori_transform = ori_transform
        self.transform=transform
        self.target_transform=target_transform
        if self.train:
            data_list = self.train_list
        else:
            data_list = self.test_list
        
        for file_name, checksum in data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f,encoding='latin1')
                self.data.append(entry['data'])
                if self.train:
                    self.filename_list.extend(entry['filenames'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  
        self._load_meta()
    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        """if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')"""
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
    
    def __getitem__(self, index:int):
        if self.train:
            img, target, filename = self.data[index], int(self.targets[index]), self.filename_list[index]
        else:
            img, target = self.data[index], int(self.targets[index])
        if self.ori_transform is not None:
            ori_img = self.ori_transform(img)
        if self.transform is not None :
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.train:
            if self.ori_transform is None:
                return img, target, filename
            else:
                return ori_img, img, target, filename
        else:
            return img, target
    def __len__(self):
        return len(self.data)
    
    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
  
 
class CIFAR100Dataset(Dataset):
    def __init__(self, root, train=True, fine_label=True, transform=True, ori_transform=None):
        if train:
            self.data,self.labels,self.filename_list=load_CIFAR_100(root,train,fine_label=fine_label)
        else:
            self.data,self.labels = load_CIFAR_100(root,train,fine_label=fine_label)
        self.transform = transform
        self.train = train
        self.origin_transform = ori_transform
    def __getitem__(self, index):
        if self.train:
            img, target, filename = self.data[index], int(self.labels[index]),self.filename_list[index]
            if self.origin_transform is not None:
                ori_img = self.origin_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            if self.origin_transform is not None:
                return ori_img, img, target, filename
            else:
                return img, target, filename
        else:
            img, target = self.data[index], int(self.labels[index])
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        return len(self.data)

def load_CIFAR_100(root, train=True, fine_label=True):
    if train:
        filename = root + 'train'
    else:
        filename = root + 'test'
 
    with open(filename, 'rb')as f:
        datadict = pickle.load(f,encoding='bytes')
 
        if train:
            # [50000, 32, 32, 3]
            X = datadict['data']
            filename_list = datadict['filenames']
            X = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1)
            Y = datadict['labels']
            Y = np.array(Y)
            return X, Y, filename_list
        else:
            # [10000, 32, 32, 3]
            X = datadict[b'data']
            filename_list = datadict[b'filenames']
            X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1)
            Y = datadict[b'fine_labels']
            Y = np.array(Y)
            return X, Y
 