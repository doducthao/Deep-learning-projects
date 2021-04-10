from lib import *
from make_datapath import make_datapath_list
from config import *

class Anno_xml():
    def __init__(self, classes):
        self.classes = classes
    
    def __call__(self, xml_path, width, height):
        # include image annotation
        ret = []

        # read file xml
        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            bndbox = list()
            name = obj.find('name').text.lower().strip()
            # print(name)
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for pt in pts:
                pixel = int(bbox.find(pt).text) - 1

                if pt == 'xmin' or pt == 'xmax':
                    pixel /= width
                else:
                    pixel /= height 
                
                bndbox.append(pixel)
            label_id = self.classes.index(name)
            bndbox.append(label_id)

            ret += [bndbox]
        return np.array(ret)


if __name__ == '__main__':
    anno_xml = Anno_xml(classes)

    root_path = './data/VOCdevkit/VOC2012'
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    idx = 1
    img_file_path = train_img_list[idx]
    print(img_file_path)

    img = cv2.imread(img_file_path)

    height, width, channels = img.shape

    annotation_infor = anno_xml(train_annotation_list[idx], width, height)
    print(annotation_infor)

