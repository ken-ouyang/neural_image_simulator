import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def init_linear(linear):
    init.ones_(linear.weight)
    if linear.bias is not None:
        linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class ExposureNet(nn.Module):
    def __init__(self):
        super(ExposureNet, self).__init__()
        linear = nn.Linear(1, 1, bias=False)
        init_linear(linear)
        self.bias = Parameter(torch.Tensor(4))
        self.bias.data.zero_()
        self.linear = linear
    
    def forward(self, input):
        # params: Estimated brightness correction in theory        
        # input: input raw (4 channels)         
        params = input[0]
        image = input[1]
        b,c,h,w = image.shape
        exp = self.linear(params)
        output = (image+ self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat((b,1,h,w))) * exp.unsqueeze(2).unsqueeze(3).repeat((1,4,h,w))
        return torch.clamp(output, 0.0, 1.0), exp
