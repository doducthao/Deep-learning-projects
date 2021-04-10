from lib import *
from config import *
from transform import ImageTransform

def make_data_path_list(phase='train'):
    rootPath = './data/hymenoptera_data/'
    target_path = osp.join(rootPath+phase+'/**/*.jpg')
    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path)
    return path_list    

def make_data_path_list_2(phase='train'):
    rootPath = './data/hymenoptera_data/'
    path_list = []
    for root, dirs, files in os.walk(osp.join(rootPath,phase)):
        for f  in files:
            if f.endswith('.jpg'):
                path_list.append(f)
    return path_list

def measureTime(a):
    start = time.time() 
    a()
    end = time.time()
    print("Time spent in {} is {}".format(a.__name__,end-start))

def train_model(net, dataloader_dict, criterior, optimizer, num_epochs):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)
    net = net.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            # if (epoch == 0) and phase == 'train':
            #     continue
            for inputs, labels in tqdm(dataloader_dict[phase]):
                # move inputs, labels to gpu/cpu
                inputs = inputs.to(device)
                labels = labels.to(device)

                # set grad of optim to be zeros
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, axis=1) # return values, indexes of maximum

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()*inputs.size(0)
                epoch_corrects += torch.sum(preds==labels) # preds == labels.data

            epoch_loss /= len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))
    torch.save(net.state_dict(), save_path)

def params_to_update(net):
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []

    update_params_1 = ['features']
    update_params_2 = ['classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias']
    update_params_3 = ['classifier.6.weight', 'classifier.6.bias']

    for name, param in net.named_parameters():
        if name in update_params_1:
            param.requires_grad = True
            params_to_update_1.append(param)
        elif name in update_params_2:
            param.requires_grad = True
            params_to_update_2.append(param)
        elif name in update_params_3:
            param.requires_grad = True
            params_to_update_3.append(param)
        else:
            param.requres_grad = False
    return params_to_update_1, params_to_update_2, params_to_update_3

def load_model(net, model_path=save_path):
    # because weights are trained on gpu colab so if you have gpu on your pc:
    if torch.cuda.is_available():
        load_weights = torch.load(model_path, map_location=torch.device('cuda:0'))
    else: # or if you dont have gpu, you need to map cuda to cpu
        load_weights = torch.load(model_path, map_location=torch.device('cpu'))
    net.load_state_dict(load_weights)
    return net