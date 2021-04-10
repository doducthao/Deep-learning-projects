from lib import *
from make_datapath import make_datapath_list
from transform import DataTransform
from extract_infor_annotation import Anno_xml
from config import *

class Mydataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, anno_xml):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_xml = anno_xml
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, ground_truth, height, width = self.pull_item(index) # ground_truth: label and bndbox of the image
        return img, ground_truth 

    def pull_item(self, index):
        img_file_path = self.img_list[index]
        img = cv2.imread(img_file_path)
        height, width, channels = img.shape

        # get anno information 
        anno_file_path = self.anno_list[index]
        anno_info = self.anno_xml(anno_file_path, width, height)
        # print(anno_file_path)
        # print(anno_info)
        # preprocessing
        img, boxes, labels = self.transform(img, self.phase, anno_info[:,:4], anno_info[:,-1])
        # print(boxes)
        # print(labels)
        # bgr -> rgb, height, width, channels -> channels, height, width
        img = torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)

        ground_truth = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, ground_truth, height, width

def collate_func(batch):
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0]) # sample[0] = img
        targets.append(torch.FloatTensor(sample[1])) # sample[1] = annotation
    
    # batch size, 3, 300, 300
    imgs = torch.stack(imgs, dim=0)

    return imgs, targets



if __name__ == "__main__":
    # prepare train, valid, annotation list
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    # prepare data transform
    train_dataset = Mydataset(train_img_list, train_annotation_list, phase='train', \
            transform = DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))
    
    val_dataset = Mydataset(val_img_list, val_annotation_list, phase='val', \
            transform = DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))

    # print(len(train_dataset))
    # print(train_dataset.__getitem__(1))

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate_func)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn = collate_func)
    dataloader_dict = {
        'train': train_dataloader,
        'val': val_dataloader
    }

    batch_iter = iter(dataloader_dict['val'])
    imgs, targets = batch_iter.next()
    print(imgs.size())
    print(len(targets))
    print(targets[0], targets[0].size())