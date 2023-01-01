import argparse, os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import torch
import torch.nn as nn
import cv2
import numpy as np
import cvbase as cvb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mmcv import ProgressBar

import util.image as im
from models import resnet_models
from dataset import FlowRefine
from util.io import load_ckpt
from edgeconnect.networks import EdgeGenerator_
from skimage.feature import canny
from PIL import Image
import torchvision.transforms.functional as FF
from util.Poisson_blend import Poisson_blend

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_threads', type=int, default=8)

    parser.add_argument('--get_mask', action='store_true')
    parser.add_argument('--output_root', type=str, default=None)

    parser.add_argument('--FIX_MASK', action='store_true')
    parser.add_argument('--DATA_ROOT', type=str,
                        default=None)
    parser.add_argument('--GT_FLOW_ROOT', type=str,
                        default=None)

    parser.add_argument('--MASK_MODE', type=str, default='bbox')
    parser.add_argument('--SAVE_FLOW', action='store_true')
    parser.add_argument('--MASK_ROOT', type=str, default=None)

    parser.add_argument('--IMAGE_SHAPE', type=int, default=[1024, 1024], nargs='+')
    parser.add_argument('--RES_SHAPE', type=int, default=[1024, 1024], nargs='+')
    parser.add_argument('--PRETRAINED', action='store_true')
    parser.add_argument('--ENLARGE_MASK', action='store_true')
    parser.add_argument('--PRETRAINED_MODEL', type=str, default=None)
    parser.add_argument('--INITIAL_HOLE', action='store_true')
    parser.add_argument('--EVAL_LIST', type=str, default=None)
    parser.add_argument('--PRINT_EVERY', type=int, default=10)

    parser.add_argument('--MASK_HEIGHT', type=int, default=120)
    parser.add_argument('--MASK_WIDTH', type=int, default=212)
    parser.add_argument('--VERTICAL_MARGIN', type=int, default=10)
    parser.add_argument('--HORIZONTAL_MARGIN', type=int, default=10)
    parser.add_argument('--MAX_DELTA_HEIGHT', type=int, default=30)
    parser.add_argument('--MAX_DELTA_WIDTH', type=int, default=53)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    test_refine_stage3(args)


def test_refine_stage3(args):
    torch.manual_seed(777)
    torch.cuda.manual_seed(777)
    args.get_mask = True
    eval_dataset = FlowRefine.FlowSeq(args, isTest=True)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=args.n_threads)
    
    dfc_resnet50 = resnet_models.Flow_Branch_Multi(input_chanels=66, NoLabels=4)
    dfc_resnet = nn.DataParallel(dfc_resnet50).cuda()
    #print('### dfc_resnet parameters:', sum(param.numel() for param in dfc_resnet.parameters()))
    dfc_resnet.eval()

    resume_iter = load_ckpt(args.PRETRAINED_MODEL,
                            [('model', dfc_resnet)], strict=False)

    print('Load Pretrained Model from', args.PRETRAINED_MODEL)
    Flow = np.empty(((480, 840, 2, 0)), dtype=np.float32)
    task_bar = ProgressBar(eval_dataset.__len__())
    
    for i, item in enumerate(eval_dataloader):
        with torch.no_grad():
            input_x = item[0].cuda()
            flow_masked = item[1].cuda()   
            gt_flow = item[2].cuda()
            mask = item[3].cuda()
            output_dir = item[4][0]        
            res_flow = dfc_resnet(input_x)
            res_flow_f = res_flow[:, :2, :, :]
            res_flow_r = res_flow[:, 2:, :, :]
            res_complete_f = res_flow_f * mask[:, 10:11, :, :] + flow_masked[:, 10:12, :, :] * (1. - mask[:, 10:11, :, :])
            res_complete_r = res_flow_r * mask[:,32:34,:,:] + flow_masked[:,32:34,:,:] * (1. - mask[:,32:34,:,:])
            '''
            res_save_f = res_complete_f[0].permute(1, 2, 0).contiguous().cpu().data.numpy()
            res_save_r = res_complete_r[0].permute(1, 2, 0).contiguous().cpu().data.numpy()
            Flow_f = np.concatenate((Flow, res_save_f[..., None]), axis=-1)
            Flow_r = np.concatenate((Flow, res_save_r[..., None]), axis=-1)
            temp_flow_masked = np.squeeze(flow_masked).permute(1, 2, 0).contiguous().cpu().data.numpy()
            temp_flow_masked = temp_flow_masked.astype(np.bool_)
            # np.set_printoptions(threshold=np.inf)
            if args.edge_guide:
                EdgeGenerator = EdgeGenerator_()
                EdgeComp_ckpt = torch.load(args.edge_completion_model)
                EdgeGenerator.load_state_dict(EdgeComp_ckpt['generator'])
                EdgeGenerator.to(torch.device('cuda:0'))
                EdgeGenerator.eval()

                # Edge completion.
                FlowF_edge = edge_completion(args, EdgeGenerator, Flow_f, temp_flow_masked[:,:,10:11], 'forward',i)
                FlowB_edge = edge_completion(args, EdgeGenerator, Flow_r, temp_flow_masked[:,:,11:12], 'backward',i)
                print('\nFinish edge completion.')
            else:
                FlowF_edge, FlowB_edge = None, None

            videoFlowF = complete_flow(args, Flow_f, temp_flow_masked[:,:,10:11], 'forward', i,FlowF_edge)
            videoFlowB = complete_flow(args, Flow_r, temp_flow_masked[:,:,11:12], 'backward', i,FlowB_edge)
            # print('videoFlowF.shape=',videoFlowF.shape)
            # path_videoFlowF = '../mid_result/videoFlowF/'
            # path_videoFlowB = '../mid_result/videoFlowB/'
            # if not os.path.exists(path_videoFlowF):
                # os.makedirs(path_videoFlowF)
            # if not os.path.exists(path_videoFlowB):
                # os.makedirs(path_videoFlowB)
            # videoFlowF_i = videoFlowF[:,:,:]
            # videoFlowF_i = flow_to_image(videoFlowF_i)    
            # videoFlowF_i_img = Image.fromarray(videoFlowF_i)
            # videoFlowF_i_img.save(path_videoFlowF + str(i) + '.png')
            # videoFlowB_i = videoFlowB[:,:,:]
            # videoFlowB_i = flow_to_image(videoFlowB_i)    
            # videoFlowB_i_img = Image.fromarray(videoFlowB_i)
            # videoFlowB_i_img.save(path_videoFlowB + str(i) + '.png')
            
            '''
            output_dir_split = output_dir.split(',') 
            output_file_f = os.path.join(args.output_root, output_dir_split[0])
            output_file_r = os.path.join(args.output_root, output_dir_split[1])
            output_basedir = os.path.dirname(output_file_f)
            if not os.path.exists(output_basedir):
                os.makedirs(output_basedir)
            res_save_f = res_complete_f[0].permute(1, 2, 0).contiguous().cpu().data.numpy()
            cvb.write_flow(res_save_f, output_file_f)
            res_save_r = res_complete_r[0].permute(1, 2, 0).contiguous().cpu().data.numpy()
            cvb.write_flow(res_save_r, output_file_r)
            task_bar.update()
    sys.stdout.write('\n')
    dfc_resnet = None
    torch.cuda.empty_cache()
    print('Refined Results Saved in', args.output_root)


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    main()