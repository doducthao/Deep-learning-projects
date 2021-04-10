from lib import *

class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-12

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.scale)
    
    def forward(self, x):
        # size = (batch, channels, height, width)
        # keepdims = True: (batch, channels, height, width) --> (batch, 1, heights, width)
        norm = x.square().sum(dim=1, keepdims=True).sqrt() + self.eps
        x = torch.div(x, norm)
        # (512,) -> (1,512) -> (1,512,1) -> (1,512,1,1)
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        return weights*x # hadaamard product
