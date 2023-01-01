# LNFVI
Local And Nonlocal Flow-Guided Video Inpainting

For the occlusion problem in the middle region, we propose a local and nonlocal optical flow video inpainting framework. First, according to the forward and backward directions of the reference frame and the sampling window, we divide the video into local and nonlocal frames, extract the local and nonlocal optical flow and feed them to the residual network for rough inpainting. Next, our approach extracts and completes the edges of the predicted flow. Finally, the composed optical flow field guides the propagation of pixels to inpaint the video content. Experimental results on DAVIS and YouTube-VOS datasets show that our method has significantly improved in terms of the image quality and optical flow quality compared with the state of the art





## Install & Requirements
The code has been tested on pytorch=1.3.0 and python3.6. Please refer to `requirements.txt` for detailed information. 

**To Install python packages**
```
conda create -n LNFVI python=3.6
conda activate LNFVI
pip install torch==1.3.0 torchvision==0.4.1
pip install -r requirements.txt
pip install imageio imageio-ffmpeg scikit-image imutils
```
**To Install flownet2 modules**
```
bash install_scripts.sh
```



## Quick start

Download [Pre Training Model]
Unzip the downloaded file and place it in the root directory of this project
Run Demo


cd tools
python video_inpaint.py --frame_dir ./demo/bear/frames --MASK_ROOT ./demo/bear/mask_bbox.png --img_size 448 896 --Propagation --PRETRAINED_MODEL ./pretrained_models/refine_network.pth --MS --th_warp 3 --FIX_MASK --edge_guide --keyword bear
