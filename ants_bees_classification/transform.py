from lib import *

class ImageTransform():
    def __init__(self,resize,mean,std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=resize,scale=(0.5,1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)]),

            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)]),

            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)