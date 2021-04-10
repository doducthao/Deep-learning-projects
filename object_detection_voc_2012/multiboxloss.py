# Jaccard
# Hard negative mining: negative default box = 3 times positive default box
# loss in regression: mse - F.SmoothL1Loss
# loss in classification: crossentropy -> F.crossentropy
from lib import *
from utils.box_utils import match
class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_threshold, negpos_ratio=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_threshold = jaccard_threshold
        self.negpos_ratio = negpos_ratio
        self.device = device
    
    def forward(self, predictions, targets):
        loc_data, conf_data, default_boxes_list = predictions

        # (batch, num_default_box, num_classes)
        num_batch = loc_data.size(0)
        num_default_box = loc_data.size(1)
        num_classes = conf_data.size(2)

        conf_target_label = torch.LongTensor(num_batch, num_default_box).to(self.device)
        loc_target = torch.Tensor(num_batch, num_default_box, 4) # 4: xmin, ymin, xmax, ymax

        for idx in range(num_batch):
            # [object: bndbox: 4 information xmin, ymin, xmax, ymax]
            truths = targets[idx][:, :-1].to(self.device) # xmin, ymin, xmax, ymax (dont take label)
            labels = targets[idx][:, -1].to(self.device) # label

            default_boxes = default_boxes_list.to(self.device)
            variances = [0.1, 0.2]
            match(self.jaccard_threshold, truths, default_boxes, variances, labels, loc_target, conf_target_label, idx)
            
        #SmoothL1Loss
        pos_mask = conf_target_label > 0 # do not get background
        # loc_data(num_batch, 8732, 4)
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data) # dim: rank 

        # positive dbox, loc_data
        loc_pos = loc_data[pos_idx].view(-1, 4)
        loc_target = loc_target[pos_idx].view(-1, 4)
        loss_loc = F.smooth_l1_loss(loc_pos, loc_target, reduction="sum")

        #loss_conf
        #CrossEntropy
        batch_conf = conf_data.view(-1, num_classes) #(num_batch*num_box, num_classes)
        loss_conf = F.cross_entropy(batch_conf, conf_target_label.view(-1), reduction="none")

        # hard negative mining
        num_pos = pos_mask.long().sum(1, keepdim=True)
        loss_conf = loss_conf.view(num_batch, -1) # torch.size([num_batch, 8732])

        _, loss_idx = loss_conf.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # idx_rank chính là thông số để biết được độ lớn loss nằm ở vị trí bao nhiêu

        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_default_box)
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        #(num_batch, 8732) -> (num_batch, 8732, 21)
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)
        conf_target_predict = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)].view(-1, num_classes) # .gt: > 
        conf_target_label_ = conf_target_label[(pos_mask+neg_mask).gt(0)]
        loss_conf = F.cross_entropy(conf_target_predict, conf_target_label_, reduction="sum")

        # total loss = loss_loc + loss_conf
        N = num_pos.sum()
        loss_loc = loss_loc/N
        loss_conf = loss_conf/N

        return loss_loc, loss_conf
        