from lib import *
from make_datapath import make_datapath_list
from config import *
from dataset import Mydataset, collate_func
from transform import DataTransform
from extract_infor_annotation import Anno_xml
from model import SSD
from multiboxloss import MultiBoxLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)
torch.backends.cudnn.benchmark = True

# dataloader
root_path = './data/VOCdevkit/VOC2012'
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(root_path)

# img_list, anno_list, phase, transform, anno_xml
train_dataset = Mydataset(train_img_list, train_anno_list, phase='train', transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))
val_dataset = Mydataset(val_img_list, val_anno_list, phase='val', transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))

train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_func)
val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate_func)

dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

net = SSD(phase='train', cfg=configs_box)
vgg_weights = torch.load('./data/weights/vgg16_reducedfc.pth')
net.vgg.load_state_dict(vgg_weights)

def weights_init(m): # m: module
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# he init
net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.confidence.apply(weights_init)

# multiboxloss
criterion = MultiBoxLoss(jaccard_threshold=0.5, negpos_ratio=3, device='cpu')

# optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

# train, valid
def train_model(net, dataloader_dict, criterion, optimizer, num_epochs):
    # move model to device
    net.to(device)

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    for epoch in range(num_epochs+1):
        time_epoch_start = time.time()
        time_iter_start = time.time()

        print('---'*20)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('---'*20)

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                print('Training')
            else:
                if (epoch+1) % 10 == 0:
                    net.eval()
                    print('---'*10)
                    print('Validation')
                else:
                    continue
            for images, targets in dataloader_dict[phase]:
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]

                # init optimizer
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase=='train'):
                    outputs = net(images)

                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == 'train':
                        loss.backward() # cal grad

                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)

                        optimizer.step() # update params

                        if iteration % 10 == 0:
                            time_iter_end = time.time()
                            duration = time_iter_end - time_iter_start
                            print('Iteration: {} || Loss: {:.4f} || 10iter: {:.4f} sec'.format(iteration, loss.item(), duration))
                            time_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()
        time_epoch_end = time.time()
        print('---'*20)
        print('Epoch {} || Epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f}'.format(epoch+1, epoch_train_loss, epoch_val_loss))
        print('Duration: {:.4f}'.format(time_epoch_end-time_epoch_start))
        time_epoch_start = time.time()

        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)

        df = pd.DataFrame(logs)
        df.to_csv('./data/ssd_logs.csv')

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        if ((epoch+1) % 10) == 0:
            torch.save(net.state_dict(), './data/weights/ssd300' + str(epoch+1) + 'pth')

num_epochs = 2
train_model(net, dataloader_dict, criterion, optimizer, num_epochs=num_epochs)