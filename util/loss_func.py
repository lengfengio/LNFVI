import torch
from torch import nn
import numpy as np
from torch import autograd


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def TVLoss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


def L1(x, y, mask=None):
    res = torch.abs(x - y)
    if mask is not None:
        res = res * mask
    return torch.mean(res)


# L.L1_mask(flow1x[:,:,:,:], gt_flow[:,10:12,:,:], mask[:,10:12,:,:])
def L1_mask(x, y, mask=None):
    res = torch.abs(x - y)
    if mask is not None:
        res = res * mask
        return torch.sum(res) / torch.sum(mask)
    return torch.mean(res)

'''

x.size,y.size = torch.size([b=16,2,240,424])   mask.size = torch.size([b=16,1,240,424])
流程：计算出res。筛选出res中对应mask>0.5的像素点作为一维array(一共包含b=16个tensor)，
进行从小到大排序，排序后每个tensor中间位置像素点的像素值作为对应res_sort列表的元素，
new_mask为mask＞0.5的元素 & res>res_sort的元素

'''
def L1_mask_hard_mining(x, y, mask):
    input_size = x.size()
    # keepdim=True 求和之后保持维度不变，不删除一维。
    # res.size = torch.size([16,1,240,424])
    res = torch.sum(torch.abs(x - y), dim=1, keepdim=True)
    with torch.no_grad():
        # idx筛选出mask＞0.5的像素点。idx为size=([16,1,240,424])的tensor,全为True和False。
        idx = mask > 0.5
        # torch.sort从小到大排序
        res_sort = [torch.sort(res[i, idx[i, ...]])[0] for i in range(idx.shape[0])]
        # print('res_sort[0].size:',res_sort[0].size())，代表res[0]中满足idx[0]的数值个数
        # .item()显示浮点型数据的更多精度
        # 新的res_sort：取res_sort列表的每一个tensor的shape[0]/2的一个对应元素作为第一个元素。
        #               新的res_sort为长度16的浮点数列表
        res_sort = [i[int(i.shape[0] * 0.5)].item() for i in res_sort]
        # .clone()使用新的地址给new_mask
        new_mask = mask.clone()
        for i in range(res.shape[0]):
            # &按位与运算符
            new_mask[i, ...] = ((mask[i, ...] > 0.5) & (res[i, ...] > res_sort[i])).float()
    res = res * new_mask
    final_res = torch.sum(res) / torch.sum(new_mask)
    # new_mask.size=torch.size([16, 1, 240, 424])
    return final_res, new_mask





def Boundary_Smoothness(x, mask):
    boundary_x = torch.abs(mask[:,:,1:,:] - mask[:,:,:-1,:])
    boundary_y = torch.abs(mask[:,:,:,1:] - mask[:,:,:,:-1])

    grad_x = torch.mean(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]), dim=1, keepdim=True)
    grad_y = torch.mean(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]), dim=1, keepdim=True)

    smoothness = torch.sum(grad_x * boundary_x) / torch.sum(boundary_x) + \
                 torch.sum(grad_y * boundary_y) / torch.sum(boundary_y)

    return smoothness


def Residual_Norm(residual):
    res = torch.sum(torch.abs(residual), dim=1)
    return torch.mean(res)


def get_gradient_x(img):
    grad_x = img[:,:,1:,:] - img[:,:,:-1,:]

    return grad_x

def get_gradient_y(img):
    grad_y = img[:,:,:,1:] - img[:,:,:,:-1]

    return grad_y


def get_flow_smoothness(fake_flow, true_flow):
    fake_grad_x = get_gradient_x(fake_flow)
    fake_grad_y = get_gradient_y(fake_flow)

    true_grad_x = get_gradient_x(true_flow)
    true_grad_y = get_gradient_y(true_flow)

    weight_x = torch.exp(-torch.mean(torch.abs(true_grad_x), dim=1, keepdim=True))
    weight_y = torch.exp(-torch.mean(torch.abs(true_grad_y), dim=1, keepdim=True))

    smoothness = torch.mean(torch.abs(fake_grad_x) * weight_x) + torch.mean(torch.abs(fake_grad_y) * weight_y)

    return smoothness

