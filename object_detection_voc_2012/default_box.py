from lib import *
from config import *

class DefaultBox():
    def __init__(self, cfg=configs_box): #cfg: see in config.py
        self.img_size = cfg['input_size']
        self.size_feature_maps = cfg['size_feature_maps']
        self.min_size = cfg['min_size']
        self.max_size = cfg['max_size']
        self.aspect_ratios = cfg['aspect_ratios']
        self.steps = cfg['steps']

    def create_default_boxes(self):
        default_boxes_list = []

        for k, f in enumerate(self.size_feature_maps):
            for i, j in itertools.product(range(f), repeat=2): # for i in range(f)...for j in range(f)
                f_k = self.img_size / self.steps[k]

                c_x = (j+0.5)/f_k
                c_y = (i+0.5)/f_k

                # small square box
                s_k  = self.min_size[k] / self.img_size # first case: 30/300
                default_boxes_list += [c_x, c_y, s_k, s_k]

                # big square box
                s_k_2 = sqrt(s_k*(self.max_size[k]/self.img_size))
                default_boxes_list += [c_x, c_y, s_k, s_k_2]

                for ar in self.aspect_ratios[k]:
                    default_boxes_list += [c_x, c_y, s_k*sqrt(ar), s_k/sqrt(ar)]
                    default_boxes_list += [c_x, c_y, s_k/sqrt(ar), s_k*sqrt(ar)]
        
        output = torch.Tensor(default_boxes_list).view(-1, 4) # each box include x center, y center, width, height
        output.clamp_(max=1, min=0) # equiv np.clip in numpy, and assign output = output.clamp(0,1)

        return output

if __name__ == "__main__":
    default_box = DefaultBox()
    default_boxes_list = default_box.create_default_boxes()
    # print(default_boxes_list)

    # print(pd.DataFrame(default_boxes_list.numpy()))

    print(default_boxes_list.shape)
