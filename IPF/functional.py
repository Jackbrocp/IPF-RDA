import torch
class totensor(object):
    def __init__(self):
        return

    def __call__(self,img):
        print(type(img))
        return torch.tensor(img)
    