import numpy as np 
import torch
from make_datapath import make_datapath_list
from extract_infor_annotation import Anno_xml
from utils.augmentation import *
from lib import *
from config import *

if __name__ == "__main__":
    # prepare train, valid, annotation list
    root_path = './data/VOCdevkit/VOC2012'
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)
    img_file_path = train_img_list[1]
    xml_file_path = train_annotation_list[1]
    print(xml_file_path)

    img = cv2.imread(img_file_path)
    height, width, channels = img.shape

    anno_xml = Anno_xml(classes)

    anno_info = anno_xml(xml_file_path, width, height)
    print(anno_info)

    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()
    current_image, current_boxes, current_labels = Compose([
                ConvertFromInts(), # convert imgs from int to float32
                ToAbsoluteCoords(), # back annotation to normal type
                PhotometricDistort(), # change color
                Expand(color_mean),
                RandomSampleCrop(),
                RandomMirror(), # rotate image 180 degrees (horizontal flip)
                ToPercentCoords(), # scale bndbox to [0-1]
                Resize(input_size),
                SubtractMeans(color_mean) # subtract mean of bgr image
                ])(img, anno_info[:,:4], anno_info[:, -1]) # expand img or not, ]img, anno_info[:,:4], anno_info[:,-1])
    print(current_boxes, current_labels)

    plt.imshow(cv2.cvtColor(current_image,cv2.COLOR_BGR2RGB))
    plt.show()



