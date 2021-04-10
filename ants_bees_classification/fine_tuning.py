from lib import *
from config import *
from utils import make_data_path_list, train_model, params_to_update
from transform import ImageTransform
from dataset import MyDataset

def main(): 
    train_list = make_data_path_list('train')
    val_list = make_data_path_list('val')

    # dataset
    train_dataset = MyDataset(train_list,transform=ImageTransform(resize, mean, std),phase='train')
    val_dataset = MyDataset(val_list,transform=ImageTransform(resize, mean, std),phase='val')

    # data loader
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size,shuffle=False)
    dataloader_dict = {'train':train_dataloader, 'val':val_dataloader}

    # net
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # loss
    criterior = nn.CrossEntropyLoss()

    # optimizer
    params1, params2, params3 = params_to_update(net)
    optimizer = optim.SGD([
        {'params':params1, 'lr':1e-4},
        {'params':params2, 'lr':5e-4},
        {'params':params3, 'lr':1e-3}], momentum = 0.9
        )
    # training
    train_model(net, dataloader_dict, criterior, optimizer, num_epochs)

if __name__ == "__main__":
    main()

