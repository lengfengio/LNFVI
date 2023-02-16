import argparse, os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import cvbase as cvb
import glob
import numpy as np
import torch
import imutils
import imageio
from PIL import Image
import cv2
import copy
import time
import scipy.ndimage
import skimage.feature 
from tools.frame_inpaint import DeepFillv1
from edgeconnect.networks import EdgeGenerator_
import torchvision.transforms
import torchvision.transforms.functional as F
from util.Poisson_blend import Poisson_blend
from util.Poisson_blend_img import Poisson_blend_img
from get_flowNN import get_flowNN
from get_flowNN_gradient import get_flowNN_gradient
from util.common_utils import flow_edge
from spatial_inpaint import spatial_inpaint
TAG_CHAR = np.array([202021.25], np.float32)
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)

def parse_argse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str,default=None)
    parser.add_argument('--FlowNet2', default=True)
    parser.add_argument('--pretrained_model_flownet2', type=str, default='./pretrained_models/FlowNet2_checkpoint.pth.tar')
    parser.add_argument('--keyword', type=str, default='bear')
    parser.add_argument('--img_size', type=int, nargs='+', default=None)
    parser.add_argument('--frame_dir', type=str, default='./demo/bear/frames')
    parser.add_argument('--DFC', default=True)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--get_mask', action='store_true')
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--output_root_Nonlocal', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--img_root', type=str,default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--DATA_ROOT', type=str,default=None)
    parser.add_argument('--DATA_Nonlocal_ROOT', type=str,default=None)
    parser.add_argument('--enlarge_mask', action='store_true')
    parser.add_argument('--enlarge_kernel', type=int,default=10)
    parser.add_argument('--rgb_max', type=float, default=255.)
    parser.add_argument('--MASK_ROOT', type=str, default='./demo/bear/mask_bbox.png')
    parser.add_argument('--data_list', type=str, default=None, help='Give the data list to extract flow')
    parser.add_argument('--FIX_MASK', action='store_true')
    parser.add_argument('--IMAGE_SHAPE', type=int, default=[240, 424], nargs='+')
    parser.add_argument('--RES_SHAPE', type=int, default=[240, 424], nargs='+')
    parser.add_argument('--GT_FLOW_ROOT', type=str,default=None)
    parser.add_argument('--PRETRAINED_MODEL', type=str, default=None)
    parser.add_argument('--INITIAL_HOLE', action='store_true')
    parser.add_argument('--Propagation', action='store_true')
    parser.add_argument('--MASK_MODE', type=str, default=None)
    parser.add_argument('--img_shape', type=int, nargs='+', default=[480, 840])
    parser.add_argument('--warp', type=int, default=5)
    parser.add_argument('--lambda1', type=float, default=0.9)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--mask_root', type=str,default=None)
    parser.add_argument('--flow_root', type=str,default=None)
    parser.add_argument('--output_root_propagation', type=str,default=None)
    parser.add_argument('--pretrained_model_inpaint', type=str,default='./pretrained_models/imagenet_deepfill.pth')
    parser.add_argument('--edge_guide', action='store_true', help='Whether use edge as guidance to complete flow')
    parser.add_argument('--edge_completion_model', default='./pretrained_models/edge_completion.pth', help="restore checkpoint")
    parser.add_argument('--nFrame')
    args = parser.parse_args()
    return args

def extract_flow(args):
    from tools.infer_flownet2 import infer
    output_file,output_Nonlocal_file = infer(args)
    flow_list = [x for x in os.listdir(output_file) if '.flo' in x]
    flow_start_no = min([int(x[:5]) for x in flow_list])
    zero_flow = cvb.read_flow(os.path.join(output_file, flow_list[0]))
    cvb.write_flow(zero_flow*0, os.path.join(output_file, '%05d.rflo' % flow_start_no))
    args.DATA_ROOT = output_file
    flow_list = [x for x in os.listdir(output_Nonlocal_file) if '.flo' in x]
    flow_start_no = min([int(x[:5]) for x in flow_list])
    zero_flow = cvb.read_flow(os.path.join(output_Nonlocal_file, flow_list[0]))
    cvb.write_flow(zero_flow*0, os.path.join(output_Nonlocal_file, '%05d.rflo' % flow_start_no))
    args.DATA_Nonlocal_ROOT = output_Nonlocal_file
def flow_completion(args):
    data_list_dir = os.path.join(args.dataset_root, 'data')
    if not os.path.exists(data_list_dir):
        os.makedirs(data_list_dir)
    args.output_root = os.path.join(args.dataset_root, 'Flow')
    args.ResNet101 = False
    from tools.test_scripts import test_refine_stage3
    from tools.test_scripts_Nonlocal import test_refine_stage3_Nonlocal
    args.PRETRAINED_MODEL = args.PRETRAINED_MODEL
    args.IMAGE_SHAPE = [480, 840]
    args.RES_SHAPE = [480, 840]
    args.DATA_ROOT = args.output_root
    args.output_root = os.path.join(args.dataset_root, 'Flow_res', 'stage3_res')
    args.output_root_Nonlocal = os.path.join(args.dataset_root, 'Flow_res', 'stage3_res_Nonlocal')
    stage3_data_list = os.path.join(data_list_dir, 'stage3_test_list.txt')
    stage3_data_list_Nonlocal = os.path.join(data_list_dir, 'stage3_test_list_Nonlocal.txt')
    from dataset.data_list import gen_flow_refine_test_mask_list,gen_flow_refine_test_mask_list_Nonlocal
    gen_flow_refine_test_mask_list(flow_root=args.DATA_ROOT,
                                   output_txt_path=stage3_data_list)
    gen_flow_refine_test_mask_list_Nonlocal(flow_root=args.DATA_Nonlocal_ROOT,
                                   output_txt_path=stage3_data_list_Nonlocal)
    args.EVAL_LIST = stage3_data_list
    args.EVAL_LIST_Nonlocal = stage3_data_list_Nonlocal
    test_refine_stage3(args)
    test_refine_stage3_Nonlocal(args)
    flow_res = './demo/'+args.keyword+'/Flow_res/stage3_res/'
    flow_res_Nonlocal = './demo/'+args.keyword+'/Flow_res/stage3_res_Nonlocal/'
    flow_result_path = './demo/'+args.keyword+'/Flow_res/flow_result/'
    if not os.path.exists(flow_result_path):
        os.makedirs(flow_result_path)
    Flow_list_f = [x for x in os.listdir(flow_res) if '.flo' in x]
    Flow_list_r = [x for x in os.listdir(flow_res) if '.rflo' in x]
    for filename in Flow_list_f:
        flow = cvb.read_flow(os.path.join(flow_res,filename))
        flow_nonlocal = cvb.read_flow(os.path.join(flow_res_Nonlocal,filename))
        flow_result = args.lambda1 * flow + args.lambda2 * flow_nonlocal
        writeFlow(os.path.join(flow_result_path, str(filename[0:5]) + '.flo'), flow_result[:, :, :])
    for filename in Flow_list_r:
        flow = cvb.read_flow(os.path.join(flow_res,filename))
        flow_nonlocal = cvb.read_flow(os.path.join(flow_res_Nonlocal,filename))
        flow_result = args.lambda1 * flow + args.lambda2 * flow_nonlocal
        writeFlow(os.path.join(flow_result_path, str(filename[0:5]) + '.rflo'), flow_result[:, :, :])
    args.flow_root = flow_result_path
def edge_inpaint(args,EdgeGenerator,mode):
    if mode not in ['forward', 'backward']:
        raise NotImplementedError
    Edge = np.empty(((480,840,0)), dtype=np.float32)
    corrFlow = np.empty(((480,840,2,0)), dtype=np.float32)
    masktemp = np.empty(((480,840,0)), dtype=np.float32)
    args.flow_root = './demo/'+args.keyword+'/Flow_res/flow_result/'
    gt_flowtoot = './demo/'+args.keyword+'/Flow/'
    if mode == 'forward':
        Flow_list = [x for x in os.listdir(args.flow_root) if '.flo' in x]
        path_flow = "./mid_result/"+args.keyword+"/forward/path_flow/"
        edge_path = "./mid_result/"+args.keyword+"/forward/edge_path/"
        edge_completed_path = "./mid_result/"+args.keyword+"/forward/edge_completed_path/"
    else:
        Flow_list = [x for x in os.listdir(args.flow_root) if '.rflo' in x]
        path_flow = "./mid_result/"+args.keyword+"/backward/path_flow/"
        edge_path = "./mid_result/"+args.keyword+"/backward/edge_path/"
        edge_completed_path = "./mid_result/"+args.keyword+"/backward/edge_completed_path/"
    Flow_list.sort()
    nFrame = len(Flow_list) + 1
    if not os.path.exists(path_flow):
        os.makedirs(path_flow)
    if not os.path.exists(edge_path):
        os.makedirs(edge_path)
    if not os.path.exists(edge_completed_path):
        os.makedirs(edge_completed_path)
    i = 0
    flow_list=[]
    mask_list=[]
    for filename in Flow_list:
        flow = cvb.read_flow(os.path.join(args.flow_root,filename))
        gt_flow = cvb.read_flow(os.path.join(gt_flowtoot,filename))
        imgH, imgW, _ = flow.shape
        flow_img = flow_to_image(flow)      
        flow_img = Image.fromarray(flow_img)
        flow_img.save(path_flow + str(i) + '.png')
        flow_img2 = flow_to_image(gt_flow)      
        flow_img2 = Image.fromarray(flow_img2)
        flow2_img = cv2.imread(path_flow + str(i) + ".png",0)  
        mask_img = cv2.imread(os.path.join("./demo",args.keyword,"mask_bbox.png"),0)
        mask_img_binary = scipy.ndimage.binary_dilation(mask_img, iterations=15)
        mask_img_binary = cv2.morphologyEx(mask_img_binary.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(np.bool_)
        mask_img_binary = scipy.ndimage.binary_fill_holes(mask_img_binary).astype(np.bool_)
        flow_img_gray = (flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2) ** 0.5
        flow_img_gray = flow_img_gray / flow_img_gray.max()
        #change
        edge = skimage.feature.canny(flow_img_gray,sigma=1,low_threshold=0.1, high_threshold=0.2)
        edge_save = edge.astype(np.uint8)
        cv2.imwrite(os.path.join(edge_path,str(i)+'.png'), (edge_save)*255)  
        edge_completed = infer(args, EdgeGenerator, torch.device('cuda:0'), flow_img_gray, edge, mask_img_binary,i)
        cv2.imwrite(os.path.join(edge_completed_path,str(i)+'.png'), edge_completed*255)
        Edge = np.concatenate((Edge, edge_completed[..., None]), axis=-1)
        corrFlow  = np.concatenate((corrFlow, flow[..., None]), axis=-1)
        masktemp  = np.concatenate((masktemp, mask_img_binary[..., None]), axis=-1)
        i = i + 1
    return Edge,corrFlow,masktemp,nFrame,imgH,imgW
def infer(args, EdgeGenerator, device, flow_img_gray, edge, mask,i):
    flow_img_gray_tensor = to_tensor(flow_img_gray)[None, :, :].float().to(device)
    edge_tensor = to_tensor(edge)[None, :, :].float().to(device)
    mask_tensor = torch.from_numpy(mask.astype(np.float64))[None, None, :, :].float().to(device)
    flow_img_gray_tensor_temp = flow_img_gray_tensor.cpu().numpy()
    flow_img_gray_tensor_temp = np.squeeze(flow_img_gray_tensor_temp)
    edge_tensor_temp = edge_tensor.cpu().numpy()
    edge_tensor_temp = np.squeeze(edge_tensor_temp)
    edges_masked = edge_tensor
    edges_masked_temp = edges_masked.cpu().numpy()
    edges_masked_temp = np.squeeze(edges_masked_temp)
    images_masked = flow_img_gray_tensor
    images_masked_temp = images_masked.cpu().numpy()
    images_masked_temp = np.squeeze(images_masked_temp)
    inputs = torch.cat((images_masked, edges_masked,mask_tensor), dim=1)
    inputs_temp = inputs.cpu().numpy()
    inputs_temp = np.squeeze(inputs_temp)
    inputs_temp = inputs_temp.reshape(inputs_temp.shape[1],inputs_temp.shape[2],inputs_temp.shape[0])
    with torch.no_grad():
        edges_completed = EdgeGenerator(inputs) # in: [grayscale(1) + edge(1) + mask(1)]
    edges_completed = edges_completed * mask_tensor + edge_tensor * (1 - mask_tensor)
    edge_completed = edges_completed[0, 0].data.cpu().numpy()
    edge_completed[edge_completed < 0.8] = 0
    edge_completed[edge_completed >= 0.8] = 1
    edges_completed_temp = np.squeeze(edge_completed)
    return edge_completed
def complete_flow(args, corrFlow, flow_mask, mode, edge,nFrame,imgH,imgW):
    if mode not in ['forward', 'backward']:
        raise NotImplementedError
    create_dir(os.path.join('mid_result/'+args.keyword+'/flow_comp', mode + '_png'))
    create_dir(os.path.join('mid_result/'+args.keyword+'/flow_comp_flo'))
    compFlow = np.zeros(((imgH, imgW, 2, nFrame)), dtype=np.float32)
    for i in range(nFrame-1):
        print("Completing {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
        flow = corrFlow[:, :, :, i]
        flow_mask_img = flow_mask[:, :, i]
        flow_mask_gradient_img = gradient_mask(flow_mask_img)
        
        if edge is not None:
            # imgH x (imgW - 1 + 1) x 2
            gradient_x = np.concatenate((np.diff(flow, axis=1), np.zeros((imgH, 1, 2), dtype=np.float32)), axis=1)
            # (imgH - 1 + 1) x imgW x 2
            gradient_y = np.concatenate((np.diff(flow, axis=0), np.zeros((1, imgW, 2), dtype=np.float32)), axis=0)

            # concatenate gradient_x and gradient_y
            gradient = np.concatenate((gradient_x, gradient_y), axis=2)

            # We can trust the gradient outside of flow_mask_gradient_img
            # We assume the gradient within flow_mask_gradient_img is 0.
            gradient[flow_mask_gradient_img, :] = 0

            # Complete the flow
            imgSrc_gy = gradient[:, :, 2 : 4]
            imgSrc_gy = imgSrc_gy[0 : imgH - 1, :, :]
            imgSrc_gx = gradient[:, :, 0 : 2]
            imgSrc_gx = imgSrc_gx[:, 0 : imgW - 1, :]              # flow: (512, 960, 2)flow 17 <---> 18
                                                                # imgSrc_gx: (512, 959, 2)
                                                                # imgSrc_gy: (511, 960, 2)
                                                                # flow_mask_img: (512, 960)
                                                                # edge[:, :, i]: (512, 960)
            compFlow[:, :, :, i] = Poisson_blend(flow, imgSrc_gx, imgSrc_gy, flow_mask_img, edge[:, :, i])

        else:
            # regionfill
            flow[:, :, 0] = rf.regionfill(flow[:, :, 0], flow_mask_img)
            flow[:, :, 1] = rf.regionfill(flow[:, :, 1], flow_mask_img)
            compFlow[:, :, :, i] = flow

        # Flow visualization. 
        flow_img = flow_to_image(compFlow[:, :, :, i])
        flow_img = Image.fromarray(flow_img)

        # Saves the flow and flow_img.
        flow_img.save(os.path.join('mid_result/'+args.keyword+'/flow_comp', mode + '_png', '%05d.png'%i))
        if mode == 'forward':
            writeFlow(os.path.join('mid_result/'+args.keyword+'/flow_comp_flo', '%05d.flo'%i), compFlow[:, :, :, i])
        else:
            writeFlow(os.path.join('mid_result/'+args.keyword+'/flow_comp_flo', '%05d.rflo'%i), compFlow[:, :, :, i])
    return compFlow

def flow_guided_propagation(args):
    deepfill_model = DeepFillv1(pretrained_model=args.pretrained_model_inpaint,
                                image_shape=args.img_shape)
    
    from tools.propagation_inpaint import propagation
    propagation(args,frame_inapint_model=deepfill_model)
def writeFlow(filename,uv,v=None):
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
def readFlow(fn):
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))
def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t


def make_colorwheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel
def gradient_mask(mask):
    gradient_mask = np.logical_or.reduce((mask,
        np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=np.bool_)), axis=0),
        np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=np.bool_)), axis=1)))

    return gradient_mask

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def main():
    args = parse_argse()
    if args.frame_dir is not None:
        args.dataset_root = os.path.dirname(args.frame_dir)
    if args.FlowNet2:
        extract_flow(args)
    if args.DFC:
        flow_completion(args)
    
    EdgeGenerator = EdgeGenerator_()
    EdgeComp_ckpt = torch.load(args.edge_completion_model)
    EdgeGenerator.load_state_dict(EdgeComp_ckpt['generator'])
    EdgeGenerator.to(torch.device('cuda:0'))
    EdgeGenerator.eval()
    print('### EdgeGenerator parameters:', sum(param.numel() for param in EdgeGenerator.parameters()))
    FlowF_edge,corrFlowF,masktempF,nFrameF,imgHF,imgWF = edge_inpaint(args,EdgeGenerator,'forward')
    FlowB_edge,corrFlowB,masktempB,nFrameB,imgHB,imgWB = edge_inpaint(args,EdgeGenerator,'backward')
    videoFlowF = complete_flow(args, corrFlowF, masktempF, 'forward', FlowF_edge,nFrameF,imgHF,imgWF)
    videoFlowB = complete_flow(args, corrFlowB, masktempB, 'backward', FlowB_edge,nFrameB,imgHB,imgWB)
    print('\nFinish flow completion.')
    # set propagation args
    assert args.mask_root is not None or args.MASK_ROOT is not None
    args.mask_root = args.MASK_ROOT if args.mask_root is None else args.mask_root
    args.img_root = args.frame_dir
    if args.output_root_propagation is None:
        args.output_root_propagation = os.path.join(args.dataset_root, 'Inpaint_Res')
    if args.img_size is not None:
        args.img_shape = args.img_size
    if args.Propagation:
        flow_guided_propagation(args)
if __name__ == '__main__':
    main()
