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


configs_box = {
    'num_classes': 21, # voc data: 20 classes + 1 background
    'input_size': 300,
    'dbox_aspect_num': [4, 6, 6, 6, 4, 4], # 4 khung hinh cho source 1, 6 khung hinh cho source 2,...
    'size_feature_maps': [38, 19, 10, 5, 3, 1],
    'steps': [8, 16, 32, 64, 100, 300], # size of default boxes
    'min_size': [30, 60, 111, 162, 213, 264],
    'max_size': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2,3], [2,3], [2,3], [2], [2]]
}

configs_model = [64, 64, 'M',
        128, 128, 'M',
        256, 256, 256, 'MC',
        512, 512, 512, 'M',
        512, 512, 512]

input_size = 300
color_mean = (104, 117, 123)
rootpath = './data/VOCdevkit/VOC2012'
batch_size = 4