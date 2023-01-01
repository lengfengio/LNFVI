import torch
import os
import random
import cv2
import cvbase as cvb
import numpy as np
import torch.utils.data as data
import util.image as im
import util.region_fill as rf
import glob
from gen_flow3 import flow_to_image3 
from gen_flow import flow_to_image      

def flow_vis(Flow_path, img_path_f, img_path_r):
    flow_f = glob.glob(os.path.join(Flow_path,'*.flo'))
    flow_f.sort()
    print(flow_f)
    for i in flow_f:
        frame_flow = cvb.read_flow(i)
        img = flow_to_image3(frame_flow)
        if not os.path.exists(img_path_f):
            os.makedirs(img_path_f)
        cv2.imwrite(os.path.join(img_path_f,i[-9:-4]+'.png'),img)
    # flow_rf = glob.glob(os.path.join(Flow_path,'*.rflo'))
    # flow_rf.sort()
    # print(flow_rf)
    # for i in flow_rf:
        # frame_flow = cvb.read_flow(i)
        # img = flow_to_image3(frame_flow)
        # if not os.path.exists(img_path_r):
            # os.makedirs(img_path_r)
        # cv2.imwrite(os.path.join(img_path_r,i[-10:-5]+'r.png'),img)
if __name__ == '__main__':
    keyword = 'surf'
    Flow_path1 = './demo/' + keyword + '/Flow'
    img_path_f1 = './flow_vis/' + keyword + '/Flow_f'
    img_path_r1 = './flow_vis/' + keyword + '/Flow_r'
    Flow_path2 = './demo/' + keyword + '/Flow_res/initial_res'
    img_path_f2 = './flow_vis/' + keyword + '/Flow_res/initial_res_f'
    img_path_r2 = './flow_vis/' + keyword + '/Flow_res/initial_res_r' 
    Flow_path3 = './demo/' + keyword + '/Flow_res/stage2_res'
    img_path_f3 = './flow_vis/' + keyword + '/Flow_res/stage2_res_f'
    img_path_r3 = './flow_vis/' + keyword + '/Flow_res/stage2_res_r'
    Flow_path4 = './demo/' + keyword + '/Flow_res/stage3_res'
    img_path_f4 = './flow_vis/' + keyword + '/Flow_res/stage3_res_f'
    img_path_r4 = './flow_vis/' + keyword + '/Flow_res/stage3_res_r'
    for i in range(1,5):
        Flow_path = eval('Flow_path' + str(i))
        img_path_f = eval('img_path_f' + str(i))
        img_path_r = eval('img_path_r' + str(i))
        flow_vis(Flow_path, img_path_f, img_path_r)