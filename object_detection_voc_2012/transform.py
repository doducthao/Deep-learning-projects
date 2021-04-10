from utils.augmentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, \
    Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans
from make_datapath import make_datapath_list
from extract_infor_annotation import Anno_xml
from lib import *

# class DataTransform:
# class DataTransform(object):
class DataTransform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(), # convert imgs from int to float32
                ToAbsoluteCoords(), # back annotation to normal type
                PhotometricDistort(), # change color
                Expand(color_mean), # expand img or not, 
                RandomSampleCrop(), # randomcrop image
                RandomMirror(), # rotate image 180 degrees (horizontal flip)
                ToPercentCoords(), # scale bndbox to [0-1]
                Resize(input_size),
                SubtractMeans(color_mean) # subtract mean of bgr image
            ]), 
            'val': Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }
    
    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


if __name__ == "__main__":
    classes = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
    ]

    # prepare train, valid, annotation list
    root_path = './data/VOCdevkit/VOC2012'
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)
    
    # read img
    img_file_path = train_img_list[0]
    img = cv2.imread(img_file_path)
    height, width, channels = img.shape

    # annotation information
    transform_anno = Anno_xml(classes)
    anno_info_list = transform_anno(train_annotation_list[0], width, height)

    # plot original img
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # prepare data transform
    color_mean = (104, 117, 123) # color mean of voc dataset
    input_size = 300 # size of SSD net
    transform = DataTransform(input_size, color_mean)

    # transform img
    phase = 'train'
    img_transformed, boxes, labels = transform(img, phase, anno_info_list[:,:4], anno_info_list[:,-1])
    img_transformed = np.clip(img_transformed, 0,1)
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()