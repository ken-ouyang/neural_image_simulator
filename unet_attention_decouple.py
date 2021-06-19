import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()

class LinearL(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        init_linear(linear)
        self.linear = linear

    def forward(self, input):
        return self.linear(input)

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = LinearL(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta

        return out

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, n_params):
        super(conv_block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm1 = AdaptiveInstanceNorm(out_ch, n_params)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm2 = AdaptiveInstanceNorm(out_ch, n_params)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, params):
        x = self.conv1(x)
        x = self.norm1(x, params)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x, params)
        x = self.relu2(x)
        
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, n_params):
        super(up_conv, self).__init__()

        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm = AdaptiveInstanceNorm(out_ch, n_params)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, params):
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x, params)
        x = self.relu(x)
        return x

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int, n_params):
        super(Attention_block, self).__init__()

        self.W_g_conv1 = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_g_norm1 = AdaptiveInstanceNorm(F_int, n_params)


        self.W_x_conv1 = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x_norm1 = AdaptiveInstanceNorm(F_int, n_params)


        self.psi_conv = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi_norm = AdaptiveInstanceNorm(1, n_params)
        self.psi_sig = nn.Sigmoid()

        self.relu1 = nn.ReLU(inplace=True)

        self.W_g_conv2 = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_g_norm2 = AdaptiveInstanceNorm(F_int, n_params)

        self.W_x_conv2 = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x_norm2 = AdaptiveInstanceNorm(F_int, n_params)

        self.global_ap = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(F_int, F_int // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(F_int // 4, F_l, kernel_size=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

        

    def forward(self, g, x, params):
        g1 = self.W_g_conv1(g)
        g1 = self.W_g_norm1(g1, params)
        x1 = self.W_x_conv1(x)
        x1 = self.W_x_norm1(x1, params)
        psi = self.relu1(g1 + x1)
        psi = self.psi_conv(psi)
        psi = self.psi_norm(psi, params)
        psi = self.psi_sig(psi)
        
        g2 = self.W_g_conv2(g)
        g2 = self.W_g_norm2(g2, params)
        x2 = self.W_x_conv2(x)
        x2 = self.W_x_norm2(x2, params)
        gap = self.global_ap(g2 + x2)
        gap = self.conv_down(gap)
        gap = self.relu2(gap)
        gap = self.conv_up(gap)
        gap = self.sig(gap)

        out = x * psi * gap
        return out


class AttenUnet_style(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, n_params, n_input_channels, n_output_channels):
        super(AttenUnet_style, self).__init__()

        self.n_params = n_params
        n_input_channels = n_input_channels + n_params

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(n_input_channels, filters[0], n_params)
        self.Conv2 = conv_block(filters[0], filters[1], n_params)
        self.Conv3 = conv_block(filters[1], filters[2], n_params)
        self.Conv4 = conv_block(filters[2], filters[3], n_params)
        self.Conv5 = conv_block(filters[3], filters[4], n_params)

        self.Up5 = up_conv(filters[4], filters[3], n_params)
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2], n_params=n_params)
        self.Up_conv5 = conv_block(filters[4], filters[3], n_params)

        self.Up4 = up_conv(filters[3], filters[2], n_params)
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1], n_params=n_params)
        self.Up_conv4 = conv_block(filters[3], filters[2], n_params)

        self.Up3 = up_conv(filters[2], filters[1], n_params)
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0], n_params=n_params)
        self.Up_conv3 = conv_block(filters[2], filters[1], n_params)

        self.Up2 = up_conv(filters[1], filters[0], n_params)
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32, n_params=n_params)
        self.Up_conv2 = conv_block(filters[1], filters[0], n_params)

        self.Conv = nn.Conv2d(filters[0], n_output_channels, kernel_size=1, stride=1, padding=0)



    def forward(self, input):

        x_param = input[0]
        x_img = input[1]
        b, c, h, w = x_img.size()
        if self.n_params == 1:
            x_param_layer = torch.unsqueeze(x_param, 2)
            x_param_layer = torch.unsqueeze(x_param_layer, 3)
            x_param_layer = x_param_layer.repeat(1, 1, h, w)
            x_input = torch.cat((x_img, x_param_layer), 1)
        else:
            x_param1 = x_param[:,0:int(self.n_params/2)]
            x_param2 = x_param[:,int(self.n_params/2):]
            x_param1_layer = torch.unsqueeze(x_param1, 2)
            x_param1_layer = torch.unsqueeze(x_param1_layer, 3)
            x_param1_layer = x_param1_layer.repeat(1, 1, h, w)
            x_param2_layer = torch.unsqueeze(x_param2, 2)
            x_param2_layer = torch.unsqueeze(x_param2_layer, 3)
            x_param2_layer = x_param2_layer.repeat(1, 1, h, w)
            x_input = torch.cat((x_img, x_param1_layer, x_param2_layer), 1)

        e1 = self.Conv1(x_input, x_param)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2, x_param)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3, x_param)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4, x_param)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5, x_param)

        d5 = self.Up5(e5, x_param)
        x4 = self.Att5(g=d5, x=e4, params=x_param)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5, x_param)

        d4 = self.Up4(d5, x_param)
        x3 = self.Att4(g=d4, x=e3, params=x_param)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4, x_param)

        d3 = self.Up3(d4, x_param)
        x2 = self.Att3(g=d3, x=e2, params=x_param)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3, x_param)

        d2 = self.Up2(d3, x_param)
        x1 = self.Att2(g=d2, x=e1, params=x_param)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2, x_param)

        out = self.Conv(d2)

        return out
