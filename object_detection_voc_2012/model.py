from lib import *
from config import *
from l2_norm import L2Norm
from default_box import DefaultBox

def create_vgg(configs=configs_model):
    layers = []
    in_channels = 3

    # M: Maxpooling, 64, 128, 512: num of filters
    
    for cfg in configs:
        if cfg == 'M': # floor
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif cfg == 'MC': # ceil
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=cfg, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)] # store inputs in RAM
            in_channels = cfg
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)

    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)
    # return nn.Sequential(*layers)

def create_extras():
    layers = []
    in_channels = 1024

    cfgs = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels=in_channels, out_channels=cfgs[0], kernel_size=1)]
    layers += [nn.Conv2d(in_channels=cfgs[0], out_channels=cfgs[1], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(in_channels=cfgs[1], out_channels=cfgs[2], kernel_size=1)]
    layers += [nn.Conv2d(in_channels=cfgs[2], out_channels=cfgs[3], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(in_channels=cfgs[3], out_channels=cfgs[4], kernel_size=1)]
    layers += [nn.Conv2d(in_channels=cfgs[4], out_channels=cfgs[5], kernel_size=3)]
    layers += [nn.Conv2d(in_channels=cfgs[5], out_channels=cfgs[6], kernel_size=1)]
    layers += [nn.Conv2d(in_channels=cfgs[6], out_channels=cfgs[7], kernel_size=3)]

    return nn.ModuleList(layers)

def create_loc_confidence(num_classes=21, bbox_ratios=[4,6,6,6,4,4]):
    # source 1,5,6 x 4, source 2,3,4 x 6
    loc_layers = []
    confidence_layers = []

    # source 1
    # loc
    loc_layers += [nn.Conv2d(512, bbox_ratios[0]*4, kernel_size=3, padding=1)] # because we have 4 offset xmin, ymin, xmax, ymax
    # confidence
    confidence_layers += [nn.Conv2d(512, bbox_ratios[0]*num_classes, kernel_size=3, padding=1)]

    # source 2
    # loc
    loc_layers += [nn.Conv2d(1024, bbox_ratios[1]*4, kernel_size=3, padding=1)]
    # confidence
    confidence_layers += [nn.Conv2d(1024, bbox_ratios[1]*num_classes, kernel_size=3, padding=1)]

    # source 3
    # loc
    loc_layers += [nn.Conv2d(512, bbox_ratios[2]*4, kernel_size=3, padding=1)]
    # confidence
    confidence_layers += [nn.Conv2d(512, bbox_ratios[2]*num_classes, kernel_size=3, padding=1)]

    # source 4
    # loc
    loc_layers += [nn.Conv2d(256, bbox_ratios[3]*4, kernel_size=3, padding=1)]
    # confidence
    confidence_layers += [nn.Conv2d(256, bbox_ratios[3]*num_classes, kernel_size=3, padding=1)]

    # source 5
    # loc
    loc_layers += [nn.Conv2d(256, bbox_ratios[4]*4, kernel_size=3, padding=1)]
    # confidence
    confidence_layers += [nn.Conv2d(256, bbox_ratios[4]*num_classes, kernel_size=3, padding=1)]

    # source 6
    # loc
    loc_layers += [nn.Conv2d(256, bbox_ratios[5]*4, kernel_size=3, padding=1)]
    # confidence
    confidence_layers += [nn.Conv2d(256, bbox_ratios[5]*num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(confidence_layers)

class SSD(nn.Module):
    def __init__(self, phase, cfg=configs_box):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']

        # create main modules
        self.vgg = create_vgg()
        self.extras = create_extras()
        self.loc, self.confidence = create_loc_confidence(cfg['num_classes'], cfg['dbox_aspect_num'])
        self.L2Norm = L2Norm()
        self.detect = Detect()

        # default box
        default_box = DefaultBox()
        self.default_boxes_list = default_box.create_default_boxes()

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        for k in range(23): # through out 23 first layers
            x = self.vgg[k](x)
        
        # source 1
        source1 = self.L2Norm(x)
        sources.append(source1)

        for k in range(23, 35): # (23, len(self.vgg))
            x = self.vgg[k](x)
        
        # source 2
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = nn.ReLU(inplace=True)(v(x)) # khong luu input vao memory, tiet kiem ram
            if k % 2 == 1: # see the SSD net image to understanding
                sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.confidence):
            # aspect ratios = 4, 6
            # (batch size, 4*aspect ratio, featuremap height, featuremap width)
            # -> (batch size, featuremap height, featuremap width, 4*aspect ratio)
            loc.append(l(x).permute(0,2,3,1).contiguous()) # ham contiguous de sap xep cac phan tu sau khi permute
            # mot cach lien tuc tren memory -> co the dung view sau nay

            conf.append(c(x).permute(0,2,3,1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], dim=1) # (batch, ...) cat (batch, ...) cat ...
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], dim=1) 

        loc = loc.view(loc.size(0), -1, 4) # (batch, 8732, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes) # (batch, 8732, 21)

        output = [loc, conf, self.default_boxes_list]
        return output

        if self.phase == "inference":
            return self.detect(output[0], output[1], output[2])


def decode(loc, default_boxes_list):
    """
    params:
        loc: [8732, 4] [delta_x, delta_y, delta_w, delta_h]
        default_boxes_list: [8732, 4] [cx_d, cy_d, w_d, h_d]
    outputs:
        boxes: [xmin, ymin, xmax, ymax]
    """

    boxes = torch.cat([
        default_boxes_list[:,:2] + 0.1*loc[:,:2],
        default_boxes_list[:,2:] * torch.exp(0.2 * loc[:,2:])
        ], dim=1) # dim = 1 to shape boxes : [cx, cy, w, h]
    
    boxes[:,:2] = boxes[:,:2] - boxes[:,2:]/2 # cal xmin, ymin
    boxes[:,2:] = boxes[:2:] + boxes[:,:2] # cal xmax, ymax

    return boxes

# non-maximum suppression
def nms(boxes, scores, overlap=0.45, top_k=200):
    """
    boxes: [num_box, xmin, ymin, xmax, ymax]
    scores: [num_box]
    """
    count = 0
    keep = scores.new(scores.size(0)).zeros_().long() # new tensor with datatype = datatype of scores, values = 0 
    
    # boxes coordinate
    x1 = boxes[:,0] # xmin
    y1 = boxes[:,1] # ymin
    x2 = boxes[:,2] # xmax
    y2 = boxes[:,3] # ymax

    # area of box
    area = torch.mul(x2-x1, y2-y1) # area of all box

    tmp_x1 = boxes.new() # tmp_x1 = tensor([])
    tmp_x2 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    _, indexes = scores.sort(0) # sort follow dim = 0
    indexes = indexes[-top_k:] # take out the last 100 elements

    while indexes.numel() > 0: # when idxes.size(0)*idxes.size(1)*... > 0
        i = indexes[-1] # id of box have max confidence

        keep[count] = i
        count += 1

        if indexes.size(0) == 1:
            break

        indexes = indexes[:-1] # *

        # information boxes
        torch.index_select(x1, 0, indexes, out=tmp_x1) # tmp_x1 = x1[indexes[-1]]
        torch.index_select(y1, 0, indexes, out=tmp_y1)
        torch.index_select(x2, 0, indexes, out=tmp_x2)
        torch.index_select(y2, 0, indexes, out=tmp_y2)

        tmp_x1 = torch.clamp(tmp_x1, min=x1[i]) # tmp_x1[tmp_x1 < x1[i]] = x1[i] | way 2: torch.clamp_(tmp_x1, min=x1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y1 = torch.clamp(tmp_x2, min=y1[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        # convert to tensor which index - 1
        tmp_w.resize_as_(tmp_x2) # must resize because of *
        tmp_h.resize_as_(tmp_y2) # or tmp_y1

        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # overlap area
        intersec = tmp_w * tmp_h

        others_area = torch.index_select(area, 0, indexes)
        union = area[i] + others_area - intersec

        iou = intersec / union # 199 iou

        indexes = indexes[iou < overlap] # keep ious < 0.45

    return keep, count

class Detect(Function): # goi class ra thi chay vao forward luon
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh # chi lay nhung bbox co conf > 0.01
        self.top_k = top_k
        self.nms_thresh = nms_thresh # giu lai nhung bbox co iou < 0.45
    
    def forward(self, loc_data, conf_data, default_boxes_list):
        num_batch = loc_data.size(0) # batch size
        num_default_box = loc_data.size(1) # 8732
        num_classes = conf_data.size(2) # 21

        # batch num, num_default_box, num_classes -> batch, num_classes, num_default_box
        conf_data = self.softmax(conf_data) 
        conf_preds = conf_data.transpose(2, 1) # transpose can only swap 2 dimension, if use permute: x=x.permute(0,2,1)

        output = torch.zeros(num_batch, num_classes, self.top_k, 5) # 5: xmin,ymin,xmax,ymax,label_id
        # process each image
        for i in range(num_batch):
            # make bboxes from ofset information and default boxes
            decode_boxes = decode(loc_data[i], default_boxes_list[i])

            # copy confidence score of the ith image
            conf_scores = conf_preds[i].clone()
            for cl in range(1, num_classes):
                c_mask = conf_scores[cl] > self.conf_thresh # take only just confidences > 0.01
                scores = conf_scores[cl][c_mask]

                if scores.nelement() == 0: # scores.numel()
                    continue
                
                # dua chieu ve giong chieu cua decode box de tinh toan  
                l_mask = c_mask.unsqueeze(-1).expand_as(decode_boxes) # (8732,4)

                boxes = decode_boxes[l_mask].view(-1,4)

                idxs, count = nms(boxes, scores, self.nms_thresh, self.top_k)

                output[i, cl, :count] = torch.cat((scores[idxs[:count]].unsqueeze(1), boxes[idxs[:count]]), 1)

        return output





if __name__ == "__main__":
    # vgg = create_vgg()
    # print(vgg)

    # extras = create_extras()
    # print(extras)

    # loc, confidence = create_loc_confidence()
    # print(loc)
    # print(confidence)

    ssd = SSD(phase='train')
    print(ssd)