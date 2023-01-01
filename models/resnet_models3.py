import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import init

affine_par = True

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads
        # print(n_dims,width,height)
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height+1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width+1, 1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        # C=512
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        # q.size= torch.Size([2, 8, 64, 841])
        # k.size= torch.Size([2, 8, 64, 841])
        # v.size= torch.Size([2, 8, 64, 841])
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)
        # print('self.rel_h + self.rel_w.size=',(self.rel_h + self.rel_w).size())
        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        # print('content_position.size=',content_position.size())
        content_position = torch.matmul(content_position, q)
        
        # print('content_content.size=',content_content.size())
        # print('content_position.size=',content_position.size())
 
        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


def down_sample(x, scalor=2, mode='bilinear'):
    if mode == 'bilinear':
        x = F.avg_pool2d(x, kernel_size=scalor, stride=scalor)
    elif mode == 'nearest':
        x = F.max_pool2d(x, kernel_size=scalor, stride=scalor)

    return x

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ExternalAttention(nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)  # 这里是对列上的数进行softmax进行归一化
        self.init_weights()  #

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S # 这里是对行进行l1Norm
        out = self.mv(attn)  # bs,n,d_model

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



# 残差网络：bottleneck残差单元模块
'''
inplanes 输入block的之前通道数
planes 在blocks中间处理时的通道数，等于输出通道数的1/4
planes * self.expansion 输出通道数
'''
class Bottleneck(nn.Module):
    # 每个stage中维度扩展的倍数
    expansion = 4
    
    # 定义初始化网络和参数   inplanes = 64    四个layer的planes分别是 64，128，256，512
    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None, heads=4, mhsa=False, resolution=None):
        super(Bottleneck, self).__init__()
        
        # 网络层
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        # dilation卷积
        if not mhsa:
            self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,
            stride=1,  # change
            padding=padding,
            bias=False,
            dilation=dilation_)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FlowBranch_Layer(nn.Module):
    def __init__(self, input_chanels, NoLabels):
        super(FlowBranch_Layer, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.upconv1 = nn.Conv2d(input_chanels, input_chanels // 2, kernel_size=3, stride=1, padding=1)

        self.upconv2 = nn.Conv2d(input_chanels // 2, 256, kernel_size=3, stride=1, padding=1)

        self.conv1_flow = nn.Conv2d(256, NoLabels, kernel_size=1, stride=1, padding=0)

        self.conv2_flow = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_flow = nn.Conv2d(128 + NoLabels, NoLabels, kernel_size=3, stride=1, padding=1)

        self.conv4_flow = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv5_flow = nn.Conv2d(64 + NoLabels, NoLabels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, input_size):
        x = self.upconv1(x)
        x = self.relu(x)
        x = F.upsample(x, (input_size[0] // 4, input_size[1] // 4), mode='bilinear', align_corners=False)
        x = self.upconv2(x)
        x = self.relu(x)

        res_4x = self.conv1_flow(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2_flow(x)
        x = self.relu(x)
        res_4x_up = F.upsample(res_4x, scale_factor=2, mode='bilinear', align_corners=False)
        conv3_input = torch.cat([x, res_4x_up], dim=1)
        res_2x = self.conv3_flow(conv3_input)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv4_flow(x)
        x = self.relu(x)
        res_2x_up = F.upsample(res_2x, scale_factor=2, mode='bilinear', align_corners=False)
        conv5_input = torch.cat([x, res_2x_up], dim=1)
        res_1x = self.conv5_flow(conv5_input)

        return res_1x, res_2x, res_4x


# 上采样模块
# FlowModule_MultiScale(ResLabels = 2048, NoLabels = 2)
class FlowModule_MultiScale(nn.Module):
    def __init__(self, input_chanels, NoLabels):
        super(FlowModule_MultiScale, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_chanels, 256, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, NoLabels, kernel_size=1, padding=0)

    def forward(self, x, res_size):
        x = F.upsample(x, (res_size[0] // 4, res_size[1] // 4), mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.relu(x)
        x = F.upsample(x, (res_size[0] // 2, res_size[1] // 2), mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = self.relu(x)
        x = F.upsample(x, res_size, mode='bilinear', align_corners=False)
        x = self.conv3(x)

        return x


class FlowModule_SingleScale(nn.Module):
    def __init__(self, input_channels, NoLabels):
        super(FlowModule_SingleScale, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, NoLabels, kernel_size=1, padding=0)

    def forward(self, x, res_size):
        x = self.conv1(x)
        x = F.upsample(x, res_size, mode='bilinear')

        return x



class ResNet(nn.Module):
    # model = ResNet(Bottleneck, [3, 4, 6, 3], input_chanels=33, NoLabels=2, FlowModule_MultiScale(ResLabels, NoLabels)) 
    # 初始化网络结构和参数
    def __init__(self, block, layers, input_chanels, NoLabels, Layer5_Module=None, resolution=(224, 224), heads=4):
        # self.inplanes为当前的fm的通道数
        self.inplanes = 64
        self.resolution = list(resolution)
        super(ResNet, self).__init__()
        
        # stem的网络层
        self.conv1 = nn.Conv2d(input_chanels, 64, kernel_size=7, stride=2, padding=3, bias=True)
        if self.conv1.stride[0] == 2:
            self.resolution[0] /= 2
        if self.conv1.stride[1] == 2:
            self.resolution[1] /= 2
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        if self.maxpool.stride == 2:
            self.resolution[0] /= 2
            self.resolution[1] /= 2
        # 64，128，256，512是扩大四倍之前的维度，即identity block的中间维度
        # stride 代表每个conv block的卷积步长
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__=4, heads=heads, mhsa=True)

        if Layer5_Module is not None:
            self.layer5 = Layer5_Module
        
        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, layers, stride=1, dilation__=1, heads=4, mhsa=False):
        downsample = None
        # 判断要不要加downsample模块，是否是Conv Block
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            # downsample模块
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par), )
        
        strides = [stride] + [1]*(layers-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.inplanes, planes, stride, dilation__, downsample, heads, mhsa, self.resolution))
            if stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)
        
    def forward(self, x):
        input_size = x.size()[2:4]
        # stem部分：conv+bn+relu+maxpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # block：四个stage
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Resnet-50后的layer5--上采样模块
        res = self.layer5(x, input_size)
        return res

    def train(self, mode=True, freezeBn=True):
        super(ResNet, self).train(mode=mode)
        if freezeBn:
            print("Freezing BatchNorm2D.")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


# Resnet101
def Flow_Branch(input_chanels=30, NoLabels=20, resolution=(224, 224), heads=4):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   input_chanels, NoLabels,
                   Layer5_Module=FlowModule_SingleScale(2048, NoLabels), resolution=resolution, heads=heads)
    return model

# Resnet50
def Flow_Branch_Multi(input_chanels=33, NoLabels=2, ResLabels=2048, resolution=(224, 448), heads=4):
    model = ResNet(Bottleneck, [3, 4, 6, 3], input_chanels, NoLabels, FlowModule_MultiScale(ResLabels, NoLabels), resolution, heads)

    return model