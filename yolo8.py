# %%
# Import các thư viện cần thiết

# Standard library imports
import os
import csv
import copy
import math
import random
import shutil
import time
from os import environ
from platform import system

# Third-party imports
import cv2
import numpy as np
import yaml
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset
from torchvision.ops import box_iou, nms

# Optional imports
try:
    import onnx
except ImportError:
    onnx = None

try:
    import albumentations
except ImportError:
    albumentations = None

try:
    from roboflow import Roboflow
except ImportError:
    os.system("pip install roboflow")
    from roboflow import Roboflow

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %%
try:
    DRIVE_SAVE_PATH = "./working/"
    os.makedirs(DRIVE_SAVE_PATH, exist_ok=True)

    SAVE_PATH = os.path.join(DRIVE_SAVE_PATH, "custom_yolo_model.pth")
    DATASET_PATH = "./wild-animals-detection-yolov8"  # Fixed path

    CHECKPOINT_DIR = os.path.join(DRIVE_SAVE_PATH, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

except:
    SAVE_PATH = "./custom_yolo_model.pth"
    DATASET_PATH = "./roboflow_dataset"
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RESUME_TRAINING = True
SAVE_CHECKPOINT_EVERY = 10

# %% [markdown]
# # KIẾN TRÚC

# %%
class Conv(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size=3,stride=1,padding=1,groups=1,activation=True):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False,groups=groups)
        self.bn=nn.BatchNorm2d(out_channels,eps=0.001,momentum=0.03)
        self.act=nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

# %%
# 2.1 Bottleneck: staack of 2 COnv with shortcut connnection (True/False)
class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,shortcut=True):
        super().__init__()
        self.conv1=Conv(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv2=Conv(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.shortcut=shortcut

    def forward(self,x):
        x_in=x # for residual connection
        x=self.conv1(x)
        x=self.conv2(x)
        if self.shortcut:
            x=x+x_in
        return x
    
# 2.2 C2f: Conv + bottleneck*N+ Conv
class C2f(nn.Module):
    def __init__(self,in_channels,out_channels, num_bottlenecks,shortcut=True):
        super().__init__()
        
        self.mid_channels=out_channels//2
        self.num_bottlenecks=num_bottlenecks

        self.conv1=Conv(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        
        # sequence of bottleneck layers
        self.m=nn.ModuleList([Bottleneck(self.mid_channels,self.mid_channels) for _ in range(num_bottlenecks)])

        self.conv2=Conv((num_bottlenecks+2)*out_channels//2,out_channels,kernel_size=1,stride=1,padding=0)
    
    def forward(self,x):
        x=self.conv1(x)

        # split x along channel dimension
        x1,x2=x[:,:x.shape[1]//2,:,:], x[:,x.shape[1]//2:,:,:]
        
        # list of outputs
        outputs=[x1,x2] # x1 is fed through the bottlenecks

        for i in range(self.num_bottlenecks):
            x1=self.m[i](x1)    # [bs,0.5c_out,w,h]
            outputs.insert(0,x1)

        outputs=torch.cat(outputs,dim=1) # [bs,0.5c_out(num_bottlenecks+2),w,h]
        out=self.conv2(outputs)

        return out

# %%
class SPPF(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=5):
        #kernel_size= size of maxpool
        super().__init__()
        hidden_channels=in_channels//2
        self.conv1=Conv(in_channels,hidden_channels,kernel_size=1,stride=1,padding=0)
        # concatenate outputs of maxpool and feed to conv2
        self.conv2=Conv(4*hidden_channels,out_channels,kernel_size=1,stride=1,padding=0)

        # maxpool is applied at 3 different sacles
        self.m=nn.MaxPool2d(kernel_size=kernel_size,stride=1,padding=kernel_size//2,dilation=1,ceil_mode=False)
    
    def forward(self,x):
        x=self.conv1(x)

        # apply maxpooling at diffent scales
        y1=self.m(x)
        y2=self.m(y1)
        y3=self.m(y2)

        # concantenate 
        y=torch.cat([x,y1,y2,y3],dim=1)

        # final conv
        y=self.conv2(y)

        return y


# %%
# backbone = DarkNet53

# return d,w,r based on version
def yolo_params(version):
    if version=='n':
        return 1/3,1/4,2.0
    elif version=='s':
        return 1/3,1/2,2.0
    elif version=='m':
        return 2/3,3/4,1.5
    elif version=='l':
        return 1.0,1.0,1.0
    elif version=='x':
        return 1.0,1.25,1.0
    
class Backbone(nn.Module):
    def __init__(self,version,in_channels=3,shortcut=True):
        super().__init__()
        d,w,r=yolo_params(version)

        # conv layers
        self.conv_0=Conv(in_channels,int(64*w),kernel_size=3,stride=2,padding=1)
        self.conv_1=Conv(int(64*w),int(128*w),kernel_size=3,stride=2,padding=1)
        self.conv_3=Conv(int(128*w),int(256*w),kernel_size=3,stride=2,padding=1)
        self.conv_5=Conv(int(256*w),int(512*w),kernel_size=3,stride=2,padding=1)
        self.conv_7=Conv(int(512*w),int(512*w*r),kernel_size=3,stride=2,padding=1)

        # c2f layers
        self.c2f_2=C2f(int(128*w),int(128*w),num_bottlenecks=int(3*d),shortcut=True)
        self.c2f_4=C2f(int(256*w),int(256*w),num_bottlenecks=int(6*d),shortcut=True)
        self.c2f_6=C2f(int(512*w),int(512*w),num_bottlenecks=int(6*d),shortcut=True)
        self.c2f_8=C2f(int(512*w*r),int(512*w*r),num_bottlenecks=int(3*d),shortcut=True)

        # sppf
        self.sppf=SPPF(int(512*w*r),int(512*w*r))
    
    def forward(self,x):
        x=self.conv_0(x)
        x=self.conv_1(x)

        x=self.c2f_2(x)

        x=self.conv_3(x)

        out1=self.c2f_4(x) # keep for output

        x=self.conv_5(out1)

        out2=self.c2f_6(x) # keep for output

        x=self.conv_7(out2)
        x=self.c2f_8(x)
        out3=self.sppf(x)

        return out1,out2,out3

print("----Nano model -----")
backbone_n=Backbone(version='n')
print(f"{sum(p.numel() for p in backbone_n.parameters())/1e6} million parameters")

print("----Small model -----")
backbone_s=Backbone(version='s')
print(f"{sum(p.numel() for p in backbone_s.parameters())/1e6} million parameters")


# %%
# upsample = nearest-neighbor interpolation with scale_factor=2
#            doesn't have trainable paramaters
class Upsample(nn.Module):
    def __init__(self,scale_factor=2,mode='nearest'):
        super().__init__()
        self.scale_factor=scale_factor
        self.mode=mode

    def forward(self,x):
        return nn.functional.interpolate(x,scale_factor=self.scale_factor,mode=self.mode)

# %%
class Neck(nn.Module):
    def __init__(self,version):
        super().__init__()
        d,w,r=yolo_params(version)

        self.up=Upsample() # no trainable parameters
        self.c2f_1=C2f(in_channels=int(512*w*(1+r)), out_channels=int(512*w),num_bottlenecks=int(3*d),shortcut=False)
        self.c2f_2=C2f(in_channels=int(768*w), out_channels=int(256*w),num_bottlenecks=int(3*d),shortcut=False)
        self.c2f_3=C2f(in_channels=int(768*w), out_channels=int(512*w),num_bottlenecks=int(3*d),shortcut=False)
        self.c2f_4=C2f(in_channels=int(512*w*(1+r)), out_channels=int(512*w*r),num_bottlenecks=int(3*d),shortcut=False)

        self.cv_1=Conv(in_channels=int(256*w),out_channels=int(256*w),kernel_size=3,stride=2, padding=1)
        self.cv_2=Conv(in_channels=int(512*w),out_channels=int(512*w),kernel_size=3,stride=2, padding=1)


    def forward(self,x_res_1,x_res_2,x):    
        # x_res_1,x_res_2,x = output of backbone
        res_1=x              # for residual connection
        
        x=self.up(x)
        x=torch.cat([x,x_res_2],dim=1)

        res_2=self.c2f_1(x)  # for residual connection
        
        x=self.up(res_2)
        x=torch.cat([x,x_res_1],dim=1)

        out_1=self.c2f_2(x)

        x=self.cv_1(out_1)

        x=torch.cat([x,res_2],dim=1)
        out_2=self.c2f_3(x)

        x=self.cv_2(out_2)

        x=torch.cat([x,res_1],dim=1)
        out_3=self.c2f_4(x)

        return out_1,out_2,out_3

# %%
# DFL
class DFL(nn.Module):
    def __init__(self,ch=16):
        super().__init__()
        
        self.ch=ch
        
        self.conv=nn.Conv2d(in_channels=ch,out_channels=1,kernel_size=1,bias=False).requires_grad_(False)
        
        # initialize conv with [0,...,ch-1]
        x=torch.arange(ch,dtype=torch.float).reshape(1,ch,1,1)
        self.conv.weight.data[:]=torch.nn.Parameter(x) # DFL only has ch parameters

    def forward(self,x):
        # x must have num_channels = 4*ch: x=[bs,4*ch,c]
        b,c,a=x.shape                           # c=4*ch
        x=x.reshape(b,4,self.ch,a).transpose(1,2)  # [bs,ch,4,a]

        # take softmax on channel dimension to get distribution probabilities
        x=x.softmax(1)                          # [b,ch,4,a]
        x=self.conv(x)                          # [b,1,4,a]
        return x.reshape(b,4,a)                    # [b,4,a]

# %%
class Head(nn.Module):
    def __init__(self,version,ch=16,num_classes=5):

        super().__init__()
        self.ch=ch                          # dfl channels
        self.coordinates=self.ch*4          # number of bounding box coordinates 
        self.nc=num_classes                 # 5 for custom dataset
        self.no=self.coordinates+self.nc    # number of outputs per anchor box

        self.stride = torch.tensor([8., 16., 32.])
        
        d,w,r=yolo_params(version=version)
        
        # for bounding boxes
        self.box=nn.ModuleList([
            nn.Sequential(Conv(int(256*w),self.coordinates,kernel_size=3,stride=1,padding=1),
                          Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1)),

            nn.Sequential(Conv(int(512*w),self.coordinates,kernel_size=3,stride=1,padding=1),
                          Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1)),

            nn.Sequential(Conv(int(512*w*r),self.coordinates,kernel_size=3,stride=1,padding=1),
                          Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1))
        ])

        # for classification
        self.cls=nn.ModuleList([
            nn.Sequential(Conv(int(256*w),self.nc,kernel_size=3,stride=1,padding=1),
                          Conv(self.nc,self.nc,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1)),

            nn.Sequential(Conv(int(512*w),self.nc,kernel_size=3,stride=1,padding=1),
                          Conv(self.nc,self.nc,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1)),

            nn.Sequential(Conv(int(512*w*r),self.nc,kernel_size=3,stride=1,padding=1),
                          Conv(self.nc,self.nc,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1))
        ])

        # dfl
        self.dfl=DFL()

    def forward(self,x):
        # x = output of Neck = list of 3 tensors with different resolution and different channel dim
        #     x[0]=[bs, ch0, w0, h0], x[1]=[bs, ch1, w1, h1], x[2]=[bs,ch2, w2, h2] 

        for i in range(len(self.box)):       # detection head i
            box=self.box[i](x[i])            # [bs,num_coordinates,w,h]
            cls=self.cls[i](x[i])            # [bs,num_classes,w,h]
            x[i]=torch.cat((box,cls),dim=1)  # [bs,num_coordinates+num_classes,w,h]

        # in training, no dfl output
        if self.training:
            return x                         # [3,bs,num_coordinates+num_classes,w,h]
        
        # in inference time, dfl produces refined bounding box coordinates
        anchors, strides = (i.transpose(0, 1) for i in self.make_anchors(x, self.stride))

        # concatenate predictions from all detection layers
        x = torch.cat([i.reshape(x[0].shape[0], self.no, -1) for i in x], dim=2) #[bs, 4*self.ch + self.nc, sum_i(h[i]w[i])]
        
        # split out predictions for box and cls
        #           box=[bs,4×self.ch,sum_i(h[i]w[i])]
        #           cls=[bs,self.nc,sum_i(h[i]w[i])]
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)


        a, b = self.dfl(box).chunk(2, 1)  # a=b=[bs,2×self.ch,sum_i(h[i]w[i])]
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)
        
        return torch.cat(tensors=(box * strides, cls.sigmoid()), dim=1)


    def make_anchors(self, x, strides, offset=0.5):
        # x= list of feature maps: x=[x[0],...,x[N-1]], in our case N= num_detection_heads=3
        #                          each having shape [bs,ch,w,h]
        #    each feature map x[i] gives output[i] = w*h anchor coordinates + w*h stride values
        
        # strides = list of stride values indicating how much 
        #           the spatial resolution of the feature map is reduced compared to the original image

        assert x is not None
        anchor_tensor, stride_tensor = [], []
        dtype, device = x[0].dtype, x[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = x[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # x coordinates of anchor centers
            sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # y coordinates of anchor centers
            sy, sx = torch.meshgrid(sy, sx)                                # all anchor centers 
            anchor_tensor.append(torch.stack((sx, sy), -1).reshape(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_tensor), torch.cat(stride_tensor)
        

# %%
import torch

# fake feature maps (bs=1, ch=3)
# ví dụ: 3 head detection tương ứng stride 8, 16, 32
x = [
    torch.zeros(1, 3, 80, 80),   # P3
    torch.zeros(1, 3, 40, 40),   # P4
    torch.zeros(1, 3, 20, 20)    # P5
]
strides = [8, 16, 32]

def make_anchors(x, strides, offset=0.5):
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')  # chú ý indexing
        anchor_tensor.append(torch.stack((sx, sy), -1).reshape(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)

# Test
anchors, strides_out = make_anchors(x, strides)

print("Anchor tensor shape:", anchors.shape)       # (8400, 2)
print("Stride tensor shape:", strides_out.shape)   # (8400, 1)

# In thử 5 anchor đầu tiên
print("First 5 anchors:\n", anchors[:5])
print("First 5 strides:\n", strides_out[:5].reshape(-1))

# In thử cuối cùng (P5)
print("Last 5 anchors:\n", anchors[-5:])
print("Last 5 strides:\n", strides_out[-5:].reshape(-1))

# %%
class MyYolo(nn.Module):
    def __init__(self, version, num_classes=5):
        super().__init__()
        self.backbone = Backbone(version=version)
        self.neck = Neck(version=version)
        self.head = Head(version=version, num_classes=num_classes)
        self.nc = num_classes

    def forward(self, x):
        x = self.backbone(x)              # return out1, out2, out3
        x = self.neck(x[0], x[1], x[2])   # return out_1, out_2, out_3
        return self.head(list(x))


# khởi tạo model với 5 class
model = MyYolo(version='n', num_classes=5)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} million parameters")

# %% [markdown]
# # UTIL

# %%
# === DEBUG UTILS ===
import torch
from contextlib import contextmanager

DEBUG_ON = False          # Bật/tắt toàn cục
DEBUG_MAX_ELEMS = 5      # In tối đa vài phần tử để đỡ rác

def tstats(name, t, mask=None):
    if not DEBUG_ON: 
        return
    try:
        if mask is not None:
            t = t[mask]
        if t.numel() == 0:
            print(f"[{name}] empty tensor")
            return
        t_det = t.detach()
        print(f"[{name}] shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
              f"min={t_det.min().item():.6f} max={t_det.max().item():.6f} "
              f"mean={t_det.float().mean().item():.6f} sum={t_det.float().sum().item():.6f} "
              f"nnz={(t_det!=0).sum().item()}/{t_det.numel()}")
        # In vài phần tử đầu
        flat = t_det.reshape(-1)
        print(f"  sample: {flat[:min(flat.numel(), DEBUG_MAX_ELEMS)].tolist()}")
        if torch.isnan(t_det).any() or torch.isinf(t_det).any():
            print(f"  WARN: {name} contains NaN/Inf")
    except Exception as e:
        print(f"[{name}] DEBUG ERROR: {e}")

def tuniq(name, t):
    if not DEBUG_ON: 
        return
    try:
        u = torch.unique(t)
        print(f"[{name}] unique({u.numel()}): {u[:min(u.numel(), DEBUG_MAX_ELEMS)].tolist()}"
              + (" ..." if u.numel() > DEBUG_MAX_ELEMS else ""))
    except Exception as e:
        print(f"[{name}] unique() error: {e}")

@contextmanager
def debug_block(title):
    if DEBUG_ON:
        print(f"\n========== DEBUG: {title} ==========")
    yield
    if DEBUG_ON:
        print(f"========== /DEBUG: {title} ==========\n")

# %%
import copy
import random
from time import time

import math
import numpy
import torch
import torchvision
from torch.nn.functional import cross_entropy

def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def export_onnx(args):
    import onnx  # noqa

    inputs = ['images']
    outputs = ['outputs']
    dynamic = {'outputs': {0: 'batch', 1: 'anchors'}}

    m = torch.load('./weights/best.pt')['model'].float()
    x = torch.zeros((1, 3, args.input_size, args.input_size))

    torch.onnx.export(m.cpu(), x.cpu(),
                      f='./weights/best.onnx',
                      verbose=False,
                      opset_version=12,
                      # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
                      do_constant_folding=True,
                      input_names=inputs,
                      output_names=outputs,
                      dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load('./weights/best.onnx')  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    onnx.save(model_onnx, './weights/best.onnx')
    # Inference example
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/autobackend.py


# def wh2xy(x):
#     y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
#     y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
#     y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
#     y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
#     y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
#     return y

def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    # Convert nx4 boxes
    # from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y

def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_tensor.append(torch.stack((sx, sy), -1).reshape(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)


def compute_metric(output, target, iou_v):
    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2, 2)
    (b1, b2) = output[:, :4].unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = intersection / (area1 + area2 - intersection)
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = numpy.zeros((output.shape[0], iou_v.shape[0]))
    correct = correct.astype(bool)
    for i in range(len(iou_v)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=output.device)


def non_max_suppression(outputs, confidence_threshold=0.001, iou_threshold=0.7):
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]  # batch size
    nc = outputs.shape[1] - 4  # number of classes
    xc = outputs[:, 4:4 + nc].amax(1) > confidence_threshold  # candidates

    # Settings
    start = time()
    limit = 0.5 + 0.05 * bs  # seconds to quit after
    output = [torch.zeros((0, 6), device=outputs.device)] * bs
    for index, x in enumerate(outputs):  # image index, image inference
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # matrix nx6 (box, confidence, cls)
        box, cls = x.split((4, nc), 1)
        box = wh2xy(box)  # (cx, cy, w, h) to (x1, y1, x2, y2)
        if nc > 1:
            i, j = (cls > confidence_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > confidence_threshold]  #Không

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes, scores
        indices = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        indices = indices[:max_det]  # limit detections

        output[index] = x[indices]
        if (time() - start) > limit:
            break  # time limit exceeded

    return output


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision
    """
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

            # Integrate area under curve
            x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap


def compute_iou(box1, box2, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
    # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    return iou - (rho2 / c2 + v * alpha)  # CIoU


def strip_optimizer(filename):
    x = torch.load(filename, map_location="cpu")
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, f=filename)


def clip_gradients(model, max_norm=10.0):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


def load_weight(model, ckpt):
    dst = model.state_dict()
    src = torch.load(ckpt)['model'].float().cpu()

    ckpt = {}
    for k, v in src.state_dict().items():
        if k in dst and v.shape == dst[k].shape:
            ckpt[k] = v

    model.load_state_dict(state_dict=ckpt, strict=False)
    return model


def set_params(model, decay):
    p1 = []
    p2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            p1.append(param)
        else:
            p2.append(param)
    return [{'params': p1, 'weight_decay': 0.00},
            {'params': p2, 'weight_decay': decay}]


def plot_lr(args, optimizer, scheduler, num_steps):
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        for i in range(num_steps):
            step = i + num_steps * epoch
            scheduler.step(step, optimizer)
            y.append(optimizer.param_groups[0]['lr'])
    print(y[0])
    print(y[-1])
    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('step')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs * num_steps)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr.png', dpi=200)
    pyplot.close()


class CosineLR:
    def __init__(self, args, params, num_steps):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 100))
        decay_steps = int(args.epochs * num_steps - warmup_steps)

        warmup_lr = numpy.linspace(min_lr, max_lr, int(warmup_steps))

        decay_lr = []
        for step in range(1, decay_steps + 1):
            alpha = math.cos(math.pi * step / decay_steps)
            decay_lr.append(min_lr + 0.5 * (max_lr - min_lr) * (1 + alpha))

        self.total_lr = numpy.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class LinearLR:
    def __init__(self, args, params, num_steps):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 100))
        decay_steps = int(args.epochs * num_steps - warmup_steps)

        warmup_lr = numpy.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        decay_lr = numpy.linspace(max_lr, min_lr, decay_steps)

        self.total_lr = numpy.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


class Assigner(torch.nn.Module):
    def __init__(self, nc=5, top_k=13, alpha=1.0, beta=6.0, eps=1E-9):
        super().__init__()
        self.top_k = top_k
        self.nc = nc
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        # with debug_block("Assigner.forward / inputs"):
        #     print(f"pd_scores: {tuple(pd_scores.shape)}  (B, A, C)")
        #     print(f"pd_bboxes: {tuple(pd_bboxes.shape)}  (B, A, 4)")
        #     print(f"anc_points: {tuple(anc_points.shape)} (A, 2)")
        #     print(f"gt_labels: {tuple(gt_labels.shape)}  (B, M, 1)")
        #     print(f"gt_bboxes: {tuple(gt_bboxes.shape)}  (B, M, 4)")
        #     print(f"mask_gt sum: {mask_gt.sum().item()}")
        #     tuniq("gt_labels uniq", gt_labels.reshape(-1))
        
        batch_size = pd_scores.size(0)
        num_max_boxes = gt_bboxes.size(1)

        if num_max_boxes == 0:
            device = gt_bboxes.device
            print("Assigner: num_max_boxes==0 -> return zeros")
            return (torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        num_anchors = anc_points.shape[0]
        shape = gt_bboxes.shape
        lt, rb = gt_bboxes.reshape(-1, 1, 4).chunk(2, 2)
        mask_in_gts = torch.cat((anc_points[None] - lt, rb - anc_points[None]), dim=2)
        mask_in_gts = mask_in_gts.reshape(shape[0], shape[1], num_anchors, -1).amin(3).gt_(self.eps)

        
        # with debug_block("in GT mask"):
        #     print("mask_in_gts sum:", mask_in_gts.sum().item())
        
        na = pd_bboxes.shape[-2]
        gt_mask = (mask_in_gts * mask_gt).bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([batch_size, num_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([batch_size, num_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, batch_size, num_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=batch_size).reshape(-1, 1).expand(-1, num_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        bbox_scores[gt_mask] = pd_scores[ind[0], :, ind[1]][gt_mask]  # b, max_num_obj, h*w

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, num_max_boxes, -1, -1)[gt_mask]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[gt_mask]
        overlaps[gt_mask] = compute_iou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)

        # with debug_block("overlaps & scores"):
        #     tstats("bbox_scores (selected)", bbox_scores[gt_mask])
        #     tstats("overlaps (selected)", overlaps[gt_mask])
        

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        
        # with debug_block("align_metric"):
        #     tstats("align_metric(all)", align_metric)
        #     tstats("align_metric(selected)", align_metric[gt_mask])

        top_k_mask = mask_gt.expand(-1, -1, self.top_k).bool()
        top_k_metrics, top_k_indices = torch.topk(align_metric, self.top_k, dim=-1, largest=True)
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(top_k_indices)
        top_k_indices.masked_fill_(~top_k_mask, 0)

        # with debug_block("top-k"):
        #     tstats("top_k_metrics", top_k_metrics)
        #     tstats("top_k_indices", top_k_indices)

        mask_top_k = torch.zeros(align_metric.shape, dtype=torch.int8, device=top_k_indices.device)
        ones = torch.ones_like(top_k_indices[:, :, :1], dtype=torch.int8, device=top_k_indices.device)
        for k in range(self.top_k):
            mask_top_k.scatter_add_(-1, top_k_indices[:, :, k:k + 1], ones)
        mask_top_k.masked_fill_(mask_top_k > 1, 0)
        mask_top_k = mask_top_k.to(align_metric.dtype)
        mask_pos = mask_top_k * mask_in_gts * mask_gt

        # with debug_block("positive mask"):
        #     print("mask_pos sum:", mask_pos.sum().item())
        #     print("pos per-gt:", mask_pos.sum(-1)[mask_gt.squeeze(-1).bool()].reshape(-1).tolist()[:20])
        
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, num_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)

        # Assigned target
        index = torch.arange(end=batch_size, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_index = target_gt_idx + index * num_max_boxes
        target_labels = gt_labels.long().flatten()[target_index]

        target_bboxes = gt_bboxes.reshape(-1, gt_bboxes.shape[-1])[target_index]

        # SỬA
        # labels hợp lệ?
        assert (target_labels >= 0).all() and (target_labels < self.nc).all(), "Assigned labels out of range"

        
        # Assigned target scores
        target_labels.clamp_(min=0, max=self.nc - 1)

        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                    #dtype=torch.int64,  #SỬA
                                    dtype=torch.float32,
                                    device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        
        # with debug_block("assigner outputs"):
        #     tstats("fg_mask", fg_mask)
        #     tstats("target_labels", target_labels)
        #     tstats("norm_align_metric", norm_align_metric)
        #     print("target_scores > 0:", (target_scores > 0).sum().item())
        
        return target_bboxes, target_scores, fg_mask.bool()


class QFL(torch.nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, outputs, targets):
        bce_loss = self.bce_loss(outputs, targets)
        return torch.pow(torch.abs(targets - outputs.sigmoid()), self.beta) * bce_loss


class VFL(torch.nn.Module):
    def __init__(self, alpha=0.75, gamma=2.00, iou_weighted=True):
        super().__init__()
        assert alpha >= 0.0
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, outputs, targets):
        assert outputs.size() == targets.size()
        targets = targets.type_as(outputs)

        if self.iou_weighted:
            focal_weight = targets * (targets > 0.0).float() + \
                           self.alpha * (outputs.sigmoid() - targets).abs().pow(self.gamma) * \
                           (targets <= 0.0).float()

        else:
            focal_weight = (targets > 0.0).float() + \
                           self.alpha * (outputs.sigmoid() - targets).abs().pow(self.gamma) * \
                           (targets <= 0.0).float()

        return self.bce_loss(outputs, targets) * focal_weight


class BoxLoss(torch.nn.Module):
    def __init__(self, dfl_ch):
        super().__init__()
        self.dfl_ch = dfl_ch

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # with debug_block("BoxLoss.forward / inputs"):
        #     print("fg_mask sum:", fg_mask.sum().item())
        #     tstats("target_scores sum per pos", torch.masked_select(target_scores.sum(-1), fg_mask))
        #     tstats("pred_bboxes(pos)", pred_bboxes[fg_mask])
        #     tstats("target_bboxes(pos)", target_bboxes[fg_mask])
            
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_box = ((1.0 - iou) * weight).sum() / target_scores_sum

        # with debug_block("IoU part"):
        #     tstats("IoU(pos)", iou)
        #     print("loss_box(partial):", float(loss_box.detach().cpu()))

        # DFL loss
        a, b = target_bboxes.chunk(2, -1)


        # SỬA: broadcast anchor_points theo batch để khớp fg_mask
        batch_size = fg_mask.shape[0]                          
        anchors_batched = anchor_points.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_anchors, 2]

        # SỬA: dùng anchors_batched thay cho anchor_points
        target = torch.cat((anchors_batched - a, b - anchors_batched), -1)
        target = target.clamp(0, self.dfl_ch - 0.01)

        # with debug_block("DFL target build"):
        #     tstats("anchor_points(pos)", anchors_batched[fg_mask])  # SỬA: dùng anchors_batched
        #     tstats("target distances(pos)", target[fg_mask])
        #     print("dfl_ch:", self.dfl_ch)
        
        loss_dfl = self.df_loss(pred_dist[fg_mask].reshape(-1, self.dfl_ch + 1), target[fg_mask])
        loss_dfl = (loss_dfl * weight).sum() / target_scores_sum


        # with debug_block("DFL final"):
        #     tstats("loss_dfl(per-pos)", loss_dfl.unsqueeze(0))
        #     print("loss_dfl:", float(loss_dfl.detach().cpu()))
        
        return loss_box, loss_dfl

    @staticmethod
    def df_loss(pred_dist, target):
        # Distribution Focal Loss (DFL)
        # https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        left_loss = cross_entropy(pred_dist, tl.reshape(-1), reduction='none').reshape(tl.shape)
        right_loss = cross_entropy(pred_dist, tr.reshape(-1), reduction='none').reshape(tl.shape)
        return (left_loss * wl + right_loss * wr).mean(-1, keepdim=True)


class ComputeLoss:
    def __init__(self, model, params):
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device

        m = model.head  # Head() module

        self.params = params
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.ch
        self.device = device

        self.box_loss = BoxLoss(m.ch - 1).to(device)
        self.cls_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.assigner = Assigner(nc=self.nc, top_k=10, alpha=0.5, beta=6.0)

        self.project = torch.arange(m.ch, dtype=torch.float, device=device)

    # def box_decode(self, anchor_points, pred_dist):
    #     b, a, c = pred_dist.shape
    #     pred_dist = pred_dist.reshape(b, a, 4, c // 4)
    #     pred_dist = pred_dist.softmax(3)
    #     pred_dist = pred_dist.matmul(self.project.type(pred_dist.dtype))
    #     lt, rb = pred_dist.chunk(2, -1)
    #     x1y1 = anchor_points - lt
    #     x2y2 = anchor_points + rb
    #     return torch.cat(tensors=(x1y1, x2y2), dim=-1)

    # def box_decode(self, anchor_points, pred_dist):
    #     b, a, c = pred_dist.shape
    #     pred_dist = pred_dist.reshape(b, a, 4, c // 4)
    #     pred_dist = pred_dist.softmax(3)
    #     # SỬA: Ensure self.project is on the same device as pred_dist
    #     project = self.project.to(pred_dist.device).type(pred_dist.dtype)
    #     pred_dist = pred_dist.matmul(project)
    #     lt, rb = pred_dist.chunk(2, -1)
    #     x1y1 = anchor_points - lt
    #     x2y2 = anchor_points + rb
    #     return torch.cat(tensors=(x1y1, x2y2), dim=-1)

    def box_decode(self, anchor_points, pred_dist):
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.reshape(b, a, 4, c // 4)
        pred_dist = pred_dist.softmax(3)
    
        # Ensure all tensors on the same device
        device = pred_dist.device
        anchor_points = anchor_points.to(device)
        project = self.project.to(device).type(pred_dist.dtype)
    
        pred_dist = pred_dist.matmul(project)
        lt, rb = pred_dist.chunk(2, -1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
    
        return torch.cat((x1y1, x2y2), dim=-1)

    # def __call__(self, outputs, targets):
    #     #*****************************
    #     # with debug_block("ComputeLoss.__call__ / inputs"):
    #     #     # Kiểm tra outputs
    #     #     print("num.feature levels:", len(outputs))
    #     #     for li, o in enumerate(outputs):
    #     #         print(f"  L{li} shape={tuple(o.shape)}")  # (B, no, H, W)
    
    #     #     # Kiểm tra targets dict
    #     #     print("targets keys:", list(targets.keys()))
    #     #     tstats("targets['idx']", targets['idx'])
    #     #     tstats("targets['cls']", targets['cls'])
    #     #     tstats("targets['box']", targets['box'])
    #     #     if 'cls' in targets:
    #     #         tuniq("targets['cls'] uniq", targets['cls'])
    #     #*****************************************




        
    #     x = torch.cat([i.reshape(outputs[0].shape[0], self.no, -1) for i in outputs], dim=2)
    #     pred_distri, pred_scores = x.split(split_size=(self.reg_max * 4, self.nc), dim=1)

    #     pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    #     pred_distri = pred_distri.permute(0, 2, 1).contiguous()



        
    #     #***********************************     
    #     # with debug_block("pred tensors"):
    #     #     tstats("pred_scores(logits)", pred_scores)
    #     #     tstats("pred_scores(sigmoid)", pred_scores.sigmoid())
    #     #     tstats("pred_distri", pred_distri)
    #     #*********************************



        
    #     data_type = pred_scores.dtype
    #     batch_size = pred_scores.shape[0]
    #     input_size = torch.tensor(outputs[0].shape[2:], device=self.device, dtype=data_type) * self.stride[0]
    #     anchor_points, stride_tensor = make_anchors(outputs, self.stride, offset=0.5)





    #     #***********************************
    #     # with debug_block("anchors"):
    #     #     tstats("anchor_points", anchor_points)
    #     #     tstats("stride_tensor", stride_tensor)
    #     #*****************************




    #     idx = targets['idx'].reshape(-1, 1)
    #     cls = targets['cls'].reshape(-1, 1)
    #     box = targets['box']



    #     #***************************
    #     # # Sanity: class hợp lệ?
    #     # assert (cls >= 0).all(), "Found negative class id"
    #     # assert (cls < self.nc).all(), f"Found class id >= nc ({self.nc})"
    #     #***************************






    #     targets = torch.cat((idx, cls, box), dim=1).to(self.device)
    #     if targets.shape[0] == 0:
    #         gt = torch.zeros(batch_size, 0, 5, device=self.device)
    #     else:
    #         i = targets[:, 0]
    #         _, counts = i.unique(return_counts=True)
    #         counts = counts.to(dtype=torch.int32)
    #         gt = torch.zeros(batch_size, counts.max(), 5, device=self.device)
    #         for j in range(batch_size):
    #             matches = i == j
    #             n = matches.sum()
    #             if n:
    #                 gt[j, :n] = targets[matches, 1:]
    #         x = gt[..., 1:5].mul_(input_size[[1, 0, 1, 0]])
    #         y = torch.empty_like(x)
    #         dw = x[..., 2] / 2  # half-width
    #         dh = x[..., 3] / 2  # half-height
    #         y[..., 0] = x[..., 0] - dw  # top left x
    #         y[..., 1] = x[..., 1] - dh  # top left y
    #         y[..., 2] = x[..., 0] + dw  # bottom right x
    #         y[..., 3] = x[..., 1] + dh  # bottom right y
    #         gt[..., 1:5] = y
    #     gt_labels, gt_bboxes = gt.split((1, 4), 2)
    #     mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

    #     #***************************
    #     # with debug_block("GT build"):
    #     #     tstats("gt_labels", gt_labels)
    #     #     tuniq("gt_labels uniq", gt_labels.reshape(-1))
    #     #     tstats("gt_bboxes", gt_bboxes)
    #     #     print("mask_gt sum:", mask_gt.sum().item())
    #     #***************************





    #     pred_bboxes = self.box_decode(anchor_points, pred_distri)


    #     #***************************
    #     # with debug_block("decoded boxes"):
    #     #     tstats("pred_bboxes(decoded)", pred_bboxes)
    #     #***************************


    #     assigned_targets = self.assigner(pred_scores.detach().sigmoid(),
    #                                      (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
    #                                      anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
    #     target_bboxes, target_scores, fg_mask = assigned_targets


    #     #***************************
    #     # with debug_block("after assigner"):
    #     #     tstats("target_bboxes", target_bboxes)
    #     #     tstats("target_scores", target_scores)
    #     #     print("target_scores > 0:", (target_scores > 0).sum().item())
    #     #     print("fg_mask sum:", fg_mask.sum().item())
    #     #***************************


    #     # # SỬA: Quan trọng: dùng clamp để đảm bảo tensor cùng device, tránh Python int.
    #     target_scores_sum = target_scores.sum().clamp(min=1.0)

    #     loss_cls = self.cls_loss(pred_scores, target_scores.to(data_type)).sum() / target_scores_sum  # BCE

    #     # #SỬA: scale lại cls loss theo số lượng foreground anchors nếu muốn
    #     # fg_mask_cls = target_scores.sum(-1) > 0
    #     # pred_scores_pos = pred_scores[fg_mask_cls]
    #     # target_scores_pos = target_scores[fg_mask_cls]
        
    #     # if fg_mask_cls.sum() > 0:
    #     #     loss_cls = self.cls_loss(pred_scores_pos, target_scores_pos).sum() / fg_mask_cls.sum()  # sum()/num_foreground
    #     # else:
    #     #     loss_cls = torch.zeros(1, device=pred_scores.device)

    #     # SỐ foreground anchors (tính theo mask) #SỬA
    #     #num_fg_anchors = fg_mask.sum().clamp(min=1.0)  
        
    #     # Classification loss (BCE) #SỬA
    #     # Dùng mean trên lớp, sau đó sum trên foreground anchors
    #     #loss_cls = (self.cls_loss(pred_scores, target_scores.to(pred_scores.dtype)).sum(dim=-1)  # sum over nc
    #                 #[fg_mask]).sum() / num_fg_anchors

    #     # with debug_block("cls loss"):
    #     #     print("target_scores_sum:", float(target_scores_sum.detach().cpu()))
    #     #     tstats("BCE elem", self.cls_loss(pred_scores, target_scores.to(data_type)))
    #     #     print("loss_cls:", float(loss_cls.detach().cpu()))

    #     # Box loss
    #     loss_box = torch.zeros(1, device=self.device)
    #     loss_dfl = torch.zeros(1, device=self.device)
    #     if fg_mask.sum():
    #         target_bboxes /= stride_tensor
    #         loss_box, loss_dfl = self.box_loss(pred_distri,
    #                                            pred_bboxes,
    #                                            anchor_points,
    #                                            target_bboxes,
    #                                            target_scores,
    #                                            target_scores_sum, fg_mask)
    #     else:
    #         print("NOTE: fg_mask.sum()==0 => loss_box=0, loss_dfl=0")


    #     # loss_box đã tính xong từ BoxLoss.forward
    #     # print("loss_box (before gain):", float(loss_box.detach().cpu()))
    #     # print("box_gain:", self.params['box'])
        
    #     loss_box *= self.params['box']  # box gain
    #     loss_cls *= self.params['cls']  # cls gain
    #     loss_dfl *= self.params['dfl']  # dfl gain

    #     # with debug_block("final losses (after gain)"):
    #     #     print(f"loss_box={float(loss_box.detach().cpu()):.6f} "
    #     #           f"loss_cls={float(loss_cls.detach().cpu()):.6f} "
    #     #           f"loss_dfl={float(loss_dfl.detach().cpu()):.6f}")
        
    #     return loss_box, loss_cls, loss_dfl

    def __call__(self, outputs, targets):
        """
        Compute YOLOv8 loss (cls + box + DFL) for a batch of predictions and targets.
        All tensors are ensured to be on the same device as outputs.
        """
        # Lấy device của model/output
        device = outputs[0].device
        data_type = outputs[0].dtype
        batch_size = outputs[0].shape[0]
    
        # reshape outputs
        x = torch.cat([i.reshape(batch_size, self.no, -1) for i in outputs], dim=2)
        pred_distri, pred_scores = x.split(split_size=(self.reg_max * 4, self.nc), dim=1)
    
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    
        # tạo anchor_points & stride_tensor
        input_size = torch.tensor(outputs[0].shape[2:], device=device, dtype=data_type) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(outputs, self.stride, offset=0.5)
        anchor_points = anchor_points.to(device)
        stride_tensor = stride_tensor.to(device)
    
        # targets -> tensor trên cùng device
        idx = targets['idx'].reshape(-1, 1)
        cls = targets['cls'].reshape(-1, 1)
        box = targets['box']
        targets = torch.cat((idx, cls, box), dim=1).to(device)
    
        # build ground-truth tensor
        if targets.shape[0] == 0:
            gt = torch.zeros(batch_size, 0, 5, device=device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            gt = torch.zeros(batch_size, counts.max(), 5, device=device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets[matches, 1:]
    
            # convert center/wh -> x1y1x2y2
            x = gt[..., 1:5].mul_(input_size[[1, 0, 1, 0]])
            y = torch.empty_like(x)
            dw = x[..., 2] / 2
            dh = x[..., 3] / 2
            y[..., 0] = x[..., 0] - dw
            y[..., 1] = x[..., 1] - dh
            y[..., 2] = x[..., 0] + dw
            y[..., 3] = x[..., 1] + dh
            gt[..., 1:5] = y
    
        gt_labels, gt_bboxes = gt.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
    
        # decode pred boxes
        pred_bboxes = self.box_decode(anchor_points, pred_distri)
    
        # assign targets
        assigned_targets = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt
        )
        target_bboxes, target_scores, fg_mask = assigned_targets
    
        # cls loss
        target_scores_sum = target_scores.sum().clamp(min=1.0)
        loss_cls = self.cls_loss(pred_scores, target_scores.to(pred_scores.dtype)).sum() / target_scores_sum
    
        # box + dfl loss
        loss_box = torch.zeros(1, device=device)
        loss_dfl = torch.zeros(1, device=device)
        if fg_mask.sum():
            target_bboxes = target_bboxes.to(device)
            target_bboxes /= stride_tensor
            loss_box, loss_dfl = self.box_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask
            )
    
        # scale losses
        loss_box *= self.params['box']
        loss_cls *= self.params['cls']
        loss_dfl *= self.params['dfl']
    
        # trả về tất cả trên device của outputs
        return loss_box.to(device), loss_cls.to(device), loss_dfl.to(device)

# %% [markdown]
# # DATASET

# %%
import math
import os
import random

import cv2
import numpy
import torch
from PIL import Image
from torch.utils import data

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'

class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, augment):
        self.params = params
        self.mosaic = augment
        self.augment = augment
        self.input_size = input_size

        # Read labels
        labels = self.load_label(filenames)
        self.labels = list(labels.values())
        self.filenames = list(labels.keys())  # update
        self.n = len(self.filenames)  # number of samples
        self.indices = range(self.n)
        # Albumentations (optional, only used if package is installed)
        self.albumentations = Albumentations()

    def __getitem__(self, index):
        index = self.indices[index]

        params = self.params
        mosaic = self.mosaic and random.random() < params['mosaic']

        if mosaic:
            # Load MOSAIC
            image, label = self.load_mosaic(index, params)
            # MixUp augmentation
            if random.random() < params['mix_up']:
                index = random.choice(self.indices)
                mix_image1, mix_label1 = image, label
                mix_image2, mix_label2 = self.load_mosaic(index, params)

                image, label = mix_up(mix_image1, mix_label1, mix_image2, mix_label2)
        else:
            # Load image
            image, shape = self.load_image(index)
            h, w = image.shape[:2]

            # Resize
            image, ratio, pad = resize(image, self.input_size, self.augment)

            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
            if self.augment:
                image, label = random_perspective(image, label, params)

        nl = len(label)  # number of labels
        h, w = image.shape[:2]
        cls = label[:, 0:1]
        box = label[:, 1:5]
        box = xy2wh(box, w, h)

        if self.augment:
            # Albumentations
            image, box, cls = self.albumentations(image, box, cls)
            nl = len(box)  # update after albumentations
            # HSV color-space
            augment_hsv(image, params)
            # Flip up-down
            if random.random() < params['flip_ud']:
                image = numpy.flipud(image)
                if nl:
                    box[:, 1] = 1 - box[:, 1]
            # Flip left-right
            if random.random() < params['flip_lr']:
                image = numpy.fliplr(image)
                if nl:
                    box[:, 0] = 1 - box[:, 0]

        target_cls = torch.zeros((nl, 1))
        target_box = torch.zeros((nl, 4))
        if nl:
            target_cls = torch.from_numpy(cls)
            target_box = torch.from_numpy(box)

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = numpy.ascontiguousarray(sample)

        return torch.from_numpy(sample), target_cls, target_box, torch.zeros(nl)

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        image = cv2.imread(self.filenames[i])
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            image = cv2.resize(image,
                               dsize=(int(w * r), int(h * r)),
                               interpolation=resample() if self.augment else cv2.INTER_LINEAR)
        return image, (h, w)

    def load_mosaic(self, index, params):
        label4 = []
        border = [-self.input_size // 2, -self.input_size // 2]
        image4 = numpy.full((self.input_size * 2, self.input_size * 2, 3), 0, dtype=numpy.uint8)
        y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b = (None, None, None, None, None, None, None, None)

        xc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))
        yc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))

        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        for i, index in enumerate(indices):
            # Load image
            image, _ = self.load_image(index)
            shape = image.shape
            if i == 0:  # top left
                x1a = max(xc - shape[1], 0)
                y1a = max(yc - shape[0], 0)
                x2a = xc
                y2a = yc
                x1b = shape[1] - (x2a - x1a)
                y1b = shape[0] - (y2a - y1a)
                x2b = shape[1]
                y2b = shape[0]
            if i == 1:  # top right
                x1a = xc
                y1a = max(yc - shape[0], 0)
                x2a = min(xc + shape[1], self.input_size * 2)
                y2a = yc
                x1b = 0
                y1b = shape[0] - (y2a - y1a)
                x2b = min(shape[1], x2a - x1a)
                y2b = shape[0]
            if i == 2:  # bottom left
                x1a = max(xc - shape[1], 0)
                y1a = yc
                x2a = xc
                y2a = min(self.input_size * 2, yc + shape[0])
                x1b = shape[1] - (x2a - x1a)
                y1b = 0
                x2b = shape[1]
                y2b = min(y2a - y1a, shape[0])
            if i == 3:  # bottom right
                x1a = xc
                y1a = yc
                x2a = min(xc + shape[1], self.input_size * 2)
                y2a = min(self.input_size * 2, yc + shape[0])
                x1b = 0
                y1b = 0
                x2b = min(shape[1], x2a - x1a)
                y2b = min(y2a - y1a, shape[0])

            pad_w = x1a - x1b
            pad_h = y1a - y1b
            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            # Labels
            label = self.labels[index].copy()
            if len(label):
                label[:, 1:] = wh2xy(label[:, 1:], shape[1], shape[0], pad_w, pad_h)
            label4.append(label)

        # Concat/clip labels
        label4 = numpy.concatenate(label4, 0)
        for x in label4[:, 1:]:
            numpy.clip(x, 0, 2 * self.input_size, out=x)

        # Augment
        image4, label4 = random_perspective(image4, label4, params, border)

        return image4, label4

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, indices = zip(*batch)

        cls = torch.cat(cls, dim=0)
        box = torch.cat(box, dim=0)

        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, dim=0)

        targets = {'cls': cls,
                   'box': box,
                   'idx': indices}
        return torch.stack(samples, dim=0), targets

    @staticmethod
    def load_label(filenames):
        x = {}
        for filename in filenames:
            try:
                # verify images
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()  # PIL verify
                shape = image.size  # image size
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert image.format.lower() in FORMATS, f'invalid image format {image.format}'

                # verify labels
                a = f'{os.sep}images{os.sep}'
                b = f'{os.sep}labels{os.sep}'
                label_path = b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt'
                if os.path.isfile(b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt'):
                    with open(label_path) as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = numpy.array(label, dtype=numpy.float32)
                    nl = len(label)
                    if nl:
                        assert (label >= 0).all()
                        assert label.shape[1] == 5
                        assert (label[:, 1:] <= 1).all()
                        _, i = numpy.unique(label, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            label = label[i]  # remove duplicates
                    else:
                        label = numpy.zeros((0, 5), dtype=numpy.float32)
                else:
                    label = numpy.zeros((0, 5), dtype=numpy.float32)
            except FileNotFoundError:
                label = numpy.zeros((0, 5), dtype=numpy.float32)
            except AssertionError:
                continue
            x[filename] = label
        return x


def wh2xy(x, w, h, pad_w=0, pad_h=0):
    # Convert nx4 boxes
    # from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def xy2wh(x, w, h):
    # warning: inplace clip
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)  # x1, x2
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)  # y1, y2

    # Convert nx4 boxes
    # from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = numpy.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def resample():
    choices = (cv2.INTER_AREA,
               cv2.INTER_CUBIC,
               cv2.INTER_LINEAR,
               cv2.INTER_NEAREST,
               cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)


def augment_hsv(image, params):
    # HSV color-space augmentation
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']

    r = numpy.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = numpy.clip(x * r[2], 0, 255).astype('uint8')

    hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def resize(image, input_size, augment):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(input_size / shape[0], input_size / shape[1])
    if not augment:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:  # resize
        image = cv2.resize(image,
                           dsize=pad,
                           interpolation=resample() if augment else cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border
    return image, (r, r), (w, h)


def candidates(box1, box2):
    # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = numpy.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)


def random_perspective(image, label, params, border=(0, 0)):
    h = image.shape[0] + border[0] * 2
    w = image.shape[1] + border[1] * 2

    # Center
    center = numpy.eye(3)
    center[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = numpy.eye(3)

    # Rotation and Scale
    rotate = numpy.eye(3)
    a = random.uniform(-params['degrees'], params['degrees'])
    s = random.uniform(1 - params['scale'], 1 + params['scale'])
    rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)
    shear[1, 0] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)

    # Translation
    translate = numpy.eye(3)
    translate[0, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * w
    translate[1, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * h

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translate @ shear @ rotate @ perspective @ center
    if (border[0] != 0) or (border[1] != 0) or (matrix != numpy.eye(3)).any():  # image changed
        image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

    # Transform label coordinates
    n = len(label)
    if n:
        xy = numpy.ones((n * 4, 3))
        xy[:, :2] = label[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ matrix.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        box = numpy.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        box[:, [0, 2]] = box[:, [0, 2]].clip(0, w)
        box[:, [1, 3]] = box[:, [1, 3]].clip(0, h)
        # filter candidates
        indices = candidates(box1=label[:, 1:5].T * s, box2=box.T)

        label = label[indices]
        label[:, 1:5] = box[indices]

    return image, label


def mix_up(image1, label1, image2, label2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    alpha = numpy.random.beta(a=32.0, b=32.0)  # mix-up ratio, alpha=beta=32.0
    image = (image1 * alpha + image2 * (1 - alpha)).astype(numpy.uint8)
    label = numpy.concatenate((label1, label2), 0)
    return image, label


class Albumentations:
    def __init__(self):
        self.transform = None
        try:
            import albumentations

            transforms = [albumentations.Blur(p=0.01),
                          albumentations.CLAHE(p=0.01),
                          albumentations.ToGray(p=0.01),
                          albumentations.MedianBlur(p=0.01)]
            self.transform = albumentations.Compose(transforms,
                                                    albumentations.BboxParams('yolo', ['class_labels']))

        except ImportError:  # package not installed, skip
            pass

    def __call__(self, image, box, cls):
        if self.transform:
            x = self.transform(image=image,
                               bboxes=box,
                               class_labels=cls)
            image = x['image']
            box = numpy.array(x['bboxes'])
            cls = numpy.array(x['class_labels'])
        return image, box, cls

# %%
# PARAMS
params = {
    'min_lr': 0.0001,
    'max_lr': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,

    # --- Tắt augmentation ---
    'hsv_h': 0.0,
    'hsv_s': 0.0,
    'hsv_v': 0.0,
    'degrees': 0.0,
    'translate': 0.0,
    'scale': 1.0,
    'shear': 0.0,
    'flip_ud': 0.0,
    'flip_lr': 0.0,
    'mosaic': 0.0,
    'mix_up': 0.0,

    # --- Dataset ---
    'nc': 5,
    'names': ['Elephant', 'Giraffe', 'Leopard', 'Lion', 'Zebra']
}

# %%
train_dir = os.path.join(DATASET_PATH, "train", "images")

filenames_train = []
for filename in os.listdir(train_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        filenames_train.append(os.path.join(train_dir, filename))

input_size = 640

# Tạo Dataset cho tập train
train_data = Dataset(
    filenames_train,
    input_size,
    params,   # đã được định nghĩa ở cell trước
    augment=False   # False = không dùng augmentation
)

# DataLoader
train_loader = DataLoader(
    train_data,
    batch_size=32,
    num_workers=2,
    pin_memory=True,
    collate_fn=Dataset.collate_fn
)

val_dir = os.path.join(DATASET_PATH, "valid", "images")

filenames_val = [os.path.join(val_dir, f) 
                 for f in os.listdir(val_dir) 
                 if f.endswith(('.jpg', '.png', '.jpeg'))]

val_dataset = Dataset(
    filenames_val,
    input_size,
    params,
    augment=False   # thường không augment validation
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,      
    num_workers=2,
    pin_memory=True,
    collate_fn=Dataset.collate_fn
)




print(f"Train_loader : {len(train_loader)} batches")

print(f"Val_loader: {len(val_loader)} batches")

# %%
batch=next(iter(train_loader))
print("All keys in batch      : ", batch[1].keys())
print(f"Input batch shape      : ", batch[0].shape)
print(f"Classification scores  : {batch[1]['cls'].shape}")
print(f"Box coordinates        : {batch[1]['box'].shape}")
print(f"Index identifier (which score belongs to which image): {batch[1]['idx'].shape}")

# %% [markdown]
# # GPU FULL

# %%
import torch
def box_decode_normalize(anchor_points, stride_tensor, pred_dist):
    b, a, c = pred_dist.shape
    pred_dist = pred_dist.reshape(b, a, 4, c // 4)
    pred_dist = pred_dist.softmax(3)
    
    # Ensure all tensors on the same device
    device = pred_dist.device
    anchor_points = anchor_points.to(device)
    project = torch.arange(16, dtype=torch.float, device=device).type(pred_dist.dtype)
    
    pred_dist = pred_dist.matmul(project)
    lt, rb = pred_dist.chunk(2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    
    return torch.cat((x1y1, x2y2), dim=-1)*stride_tensor/input_size

# def bbox_iou_torch(box1, box2):
#     """
#     Compute IoU between two sets of boxes using PyTorch (GPU compatible)
#     box1: (N,4) xyxy
#     box2: (M,4) xyxy
#     return: (N,M) IoU matrix
#     """
#     N = box1.shape[0]
#     M = box2.shape[0]

#     inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
#     inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
#     inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
#     inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

#     inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
#     inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
#     inter_area = inter_w * inter_h

#     area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
#     area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
#     union = area1[:, None] + area2[None, :] - inter_area

#     iou = inter_area / (union + 1e-16)
#     return iou


# def match_detections_torch(pred_bboxes, pred_conf, pred_classes, gt_bboxes, gt_classes, iou_threshold=0.5):
#     """
#     GPU version of TP matching
#     """
#     device = pred_bboxes.device
#     num_pred = pred_bboxes.shape[0]
#     tp = torch.zeros((num_pred,), device=device)

#     if gt_bboxes.shape[0] == 0 or num_pred == 0:
#         return tp

#     sort_idx = torch.argsort(-pred_conf)
#     pred_bboxes = pred_bboxes[sort_idx]
#     pred_classes = pred_classes[sort_idx]

#     assigned_gt = torch.zeros(gt_bboxes.shape[0], dtype=torch.bool, device=device)
#     ious = bbox_iou_torch(pred_bboxes, gt_bboxes)

#     for i in range(num_pred):
#         cls_matches = (pred_classes[i] == gt_classes)
#         iou_matches = ious[i] >= iou_threshold
#         matches = cls_matches & iou_matches & (~assigned_gt)
#         if matches.any():
#             gt_idx = torch.argmax(ious[i] * matches.float())
#             tp[sort_idx[i]] = 1
#             assigned_gt[gt_idx] = True

#     return tp


# def build_tp_matrix_torch(pred_bboxes, pred_conf, pred_classes, gt_bboxes, gt_classes):
#     """
#     Trả về ma trận TP (num_preds, 10) trên GPU
#     """
#     device = pred_bboxes.device
#     iou_thresholds = torch.arange(0.5, 1.0, 0.05, device=device)
#     num_preds = pred_bboxes.shape[0]
#     tp_matrix = torch.zeros((num_preds, len(iou_thresholds)), device=device)

#     for j, thr in enumerate(iou_thresholds):
#         tp_matrix[:, j] = match_detections_torch(
#             pred_bboxes, pred_conf, pred_classes, gt_bboxes, gt_classes, iou_threshold=thr
#         )

#     return tp_matrix


# %%
# import torch

# def bbox_iou_torch(boxes1, boxes2):
#     """
#     boxes1: (B, N, 4) [x1, y1, x2, y2]
#     boxes2: (B, M, 4)
#     return: (B, N, M)
#     """
#     # (B, N, 1, 4), (B, 1, M, 4) -> broadcast
#     b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[..., 0:1], boxes1[..., 1:2], boxes1[..., 2:3], boxes1[..., 3:4]
#     b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[..., 0], boxes2[..., 1], boxes2[..., 2], boxes2[..., 3]

#     inter_x1 = torch.max(b1_x1, b2_x1[:, None, :])
#     inter_y1 = torch.max(b1_y1, b2_y1[:, None, :])
#     inter_x2 = torch.min(b1_x2, b2_x2[:, None, :])
#     inter_y2 = torch.min(b1_y2, b2_y2[:, None, :])

#     inter_w = (inter_x2 - inter_x1).clamp(min=0)
#     inter_h = (inter_y2 - inter_y1).clamp(min=0)
#     inter_area = inter_w * inter_h

#     area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
#     area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

#     union = area1[:, :, None] + area2[:, None, :] - inter_area
#     return inter_area / (union + 1e-6)


# def build_tp_matrix_batch(pred_bboxes, pred_conf, pred_classes,
#                           gt_bboxes, gt_classes, B=32):
#     """
#     Batch version, hỗ trợ input flatten (B*N, 4)

#     pred_bboxes: (B*N, 4) hoặc (B, N, 4)
#     pred_conf:   (B*N,)   hoặc (B, N)
#     pred_classes:(B*N,)   hoặc (B, N)
#     gt_bboxes:   (B, M, 4)
#     gt_classes:  (B, M)

#     return: tp_matrix (B*N, T) với T=10 thresholds (0.5:0.05:0.95)
#     """
#     device = pred_bboxes.device
#     iou_thresholds = torch.arange(0.5, 1.0, 0.05, device=device)  # (T,)

#     # Nếu input đã flatten → reshape lại
#     if pred_bboxes.dim() == 2:   # (B*N, 4)
#         assert B is not None, "Cần truyền số batch B nếu input flatten"
#         N = pred_bboxes.shape[0] // B
#         pred_bboxes = pred_bboxes.view(B, N, 4)
#         pred_conf   = pred_conf.view(B, N)
#         pred_classes= pred_classes.view(B, N)

#     B, N, _ = pred_bboxes.shape
#     M = gt_bboxes.shape[1]
#     T = len(iou_thresholds)

#     # Sort predictions by confidence
#     sort_idx = torch.argsort(pred_conf, dim=1, descending=True)
#     pred_bboxes = torch.gather(pred_bboxes, 1, sort_idx[..., None].expand(-1, -1, 4))
#     pred_classes = torch.gather(pred_classes, 1, sort_idx)

#     # IoU (B, N, M)
#     ious = bbox_iou_torch(pred_bboxes, gt_bboxes)

#     tp_matrix = torch.zeros((B, N, T), device=device)

#     for b in range(B):
#         assigned_gt = torch.zeros(M, dtype=torch.bool, device=device)
#         for n in range(N):
#             cls_matches = (pred_classes[b, n] == gt_classes[b])  # (M,)
#             for t, thr in enumerate(iou_thresholds):
#                 iou_matches = (ious[b, n] >= thr)
#                 matches = cls_matches & iou_matches & (~assigned_gt)
#                 if matches.any():
#                     gt_idx = torch.argmax(ious[b, n] * matches.float())
#                     tp_matrix[b, n, t] = 1
#                     assigned_gt[gt_idx] = True

#     # Flatten về (B*N, T)
#     tp_matrix = tp_matrix.view(B * N, T)
#     return tp_matrix

# %%
# import torch

# def bbox_iou_torch(boxes1, boxes2):
#     """
#     boxes1: (B, N, 4) [x1, y1, x2, y2]
#     boxes2: (B, M, 4)
#     return: (B, N, M)
#     """
#     b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[..., 0:1], boxes1[..., 1:2], boxes1[..., 2:3], boxes1[..., 3:4]
#     b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[..., 0:1], boxes2[..., 1:2], boxes2[..., 2:3], boxes2[..., 3:4]

#     # (B, N, 1) vs (B, 1, M) → broadcast thành (B, N, M)
#     inter_x1 = torch.max(b1_x1, b2_x1.transpose(1, 2))
#     inter_y1 = torch.max(b1_y1, b2_y1.transpose(1, 2))
#     inter_x2 = torch.min(b1_x2, b2_x2.transpose(1, 2))
#     inter_y2 = torch.min(b1_y2, b2_y2.transpose(1, 2))

#     inter_w = (inter_x2 - inter_x1).clamp(min=0)
#     inter_h = (inter_y2 - inter_y1).clamp(min=0)
#     inter_area = inter_w * inter_h

#     area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)       # (B, N, 1)
#     area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)       # (B, M, 1)

#     union = area1 + area2.transpose(1, 2) - inter_area
#     return inter_area / (union + 1e-6)

# def pack_targets(gt_bboxes, gt_classes, gt_idx, B, pad_val=-1):
#     """
#     Gom ground truth về dạng (B, M, 4), (B, M)
#     gt_bboxes: (T, 4)
#     gt_classes: (T,)
#     gt_idx: (T,) batch index ứng với mỗi GT
#     B: batch size
#     pad_val: giá trị điền vào chỗ trống (default -1)
#     """
#     # Số lượng box tối đa trên 1 ảnh
#     M = gt_idx.long().bincount(minlength=B).max().item()

#     # Khởi tạo tensor
#     out_boxes = torch.full((B, M, 4), pad_val, device=gt_bboxes.device, dtype=gt_bboxes.dtype)
#     out_classes = torch.full((B, M), pad_val, device=gt_classes.device, dtype=gt_classes.dtype)

#     counter = torch.zeros(B, dtype=torch.long, device=gt_bboxes.device)

#     for i in range(gt_bboxes.shape[0]):
#         b = int(gt_idx[i])
#         j = counter[b].item()
#         out_boxes[b, j] = gt_bboxes[i]
#         out_classes[b, j] = gt_classes[i]
#         counter[b] += 1

#     return out_boxes, out_classes


# def build_tp_matrix_batch(pred_bboxes, pred_conf, pred_classes,
#                           gt_bboxes, gt_classes, gt_idx=None, B=32):
#     """
#     Batch version, hỗ trợ cả input flatten (B*N, 4) và target flatten (T, 4)

#     pred_bboxes: (B*N, 4) hoặc (B, N, 4)
#     pred_conf:   (B*N,)   hoặc (B, N)
#     pred_classes:(B*N,)   hoặc (B, N)
#     gt_bboxes:   (B, M, 4) hoặc (T, 4)
#     gt_classes:  (B, M)   hoặc (T,)
#     gt_idx:      (T,) nếu gt flatten
#     B: batch size

#     return: tp_matrix (B*N, T) với T=10 thresholds (0.5:0.05:0.95)
#     """
#     device = pred_bboxes.device
#     iou_thresholds = torch.arange(0.5, 1.0, 0.05, device=device)  # (T,)

#     # Nếu pred flatten → reshape
#     if pred_bboxes.dim() == 2:   # (B*N, 4)
#         assert B is not None, "Cần truyền số batch B nếu input flatten"
#         N = pred_bboxes.shape[0] // B
#         pred_bboxes = pred_bboxes.view(B, N, 4)
#         pred_conf   = pred_conf.view(B, N)
#         pred_classes= pred_classes.view(B, N)

#     B, N, _ = pred_bboxes.shape

#     # Nếu gt flatten → pack về (B, M, 4), (B, M)
#     if gt_bboxes.dim() == 2:   # (T, 4)
#         assert gt_idx is not None, "Cần gt_idx để pack target"
#         gt_bboxes, gt_classes = pack_targets(gt_bboxes, gt_classes, gt_idx, B)

#     M = gt_bboxes.shape[1]
#     T = len(iou_thresholds)

#     # Sort predictions by confidence
#     sort_idx = torch.argsort(pred_conf, dim=1, descending=True)
#     pred_bboxes = torch.gather(pred_bboxes, 1, sort_idx[..., None].expand(-1, -1, 4))
#     pred_classes = torch.gather(pred_classes, 1, sort_idx)

#     # IoU (B, N, M)
#     ious = bbox_iou_torch(pred_bboxes, gt_bboxes)

#     tp_matrix = torch.zeros((B, N, T), device=device)

#     for b in range(B):
#         assigned_gt = torch.zeros(M, dtype=torch.bool, device=device)
#         for n in range(N):
#             cls_matches = (pred_classes[b, n] == gt_classes[b])  # (M,)
#             for t, thr in enumerate(iou_thresholds):
#                 iou_matches = (ious[b, n] >= thr)
#                 matches = cls_matches & iou_matches & (~assigned_gt)
#                 if matches.any():
#                     gt_idx_ = torch.argmax(ious[b, n] * matches.float())
#                     tp_matrix[b, n, t] = 1
#                     assigned_gt[gt_idx_] = True

#     # Flatten về (B*N, T)
#     tp_matrix = tp_matrix.view(B * N, T)
#     return tp_matrix


# %%
import torch

def pack_targets(gt_bboxes, gt_classes, gt_idx, B, pad_val=-1):
    """
    Gom ground truth về dạng (B, M, 4), (B, M)
    
    gt_bboxes: (T, 4)
    gt_classes: (T,)
    gt_idx: (T,) batch index của mỗi GT
    B: batch size
    pad_val: giá trị điền vào chỗ trống (default -1)
    """
    # Số lượng box tối đa trên 1 ảnh
    M = gt_idx.long().bincount(minlength=B).max().item()

    # Khởi tạo tensor
    out_boxes = torch.full((B, M, 4), pad_val, device=gt_bboxes.device, dtype=gt_bboxes.dtype)
    out_classes = torch.full((B, M), pad_val, device=gt_classes.device, dtype=gt_classes.dtype)

    counter = torch.zeros(B, dtype=torch.long, device=gt_bboxes.device)

    for i in range(gt_bboxes.shape[0]):
        b = int(gt_idx[i])
        j = counter[b].item()
        out_boxes[b, j] = gt_bboxes[i]
        out_classes[b, j] = gt_classes[i]
        counter[b] += 1

    return out_boxes, out_classes
def bbox_iou_torch(boxes1, boxes2):
    """
    boxes1: (B, N, 4) [x1, y1, x2, y2]
    boxes2: (B, M, 4)
    return: (B, N, M)
    """
    B, N, _ = boxes1.shape
    _, M, _ = boxes2.shape

    # broadcast (B, N, 1, 4) vs (B, 1, M, 4)
    b1 = boxes1.unsqueeze(2)  # (B, N, 1, 4)
    b2 = boxes2.unsqueeze(1)  # (B, 1, M, 4)

    inter_x1 = torch.max(b1[...,0], b2[...,0])
    inter_y1 = torch.max(b1[...,1], b2[...,1])
    inter_x2 = torch.min(b1[...,2], b2[...,2])
    inter_y2 = torch.min(b1[...,3], b2[...,3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[...,2]-boxes1[...,0]) * (boxes1[...,3]-boxes1[...,1])
    area2 = (boxes2[...,2]-boxes2[...,0]) * (boxes2[...,3]-boxes2[...,1])
    union = area1.unsqueeze(-1) + area2.unsqueeze(1) - inter_area

    return inter_area / (union + 1e-6)

def build_tp_matrix_batch_yolov8_tensor(pred_bboxes, pred_conf, pred_classes,
                                        gt_bboxes, gt_classes, B=None):
    """
    Vectorized (no Python loops over N/T) — YOLOv8-style.

    Inputs:
      pred_bboxes: (B*N, 4) or (B, N, 4)
      pred_conf:   (B*N,)   or (B, N)
      pred_classes:(B*N,)   or (B, N)
      gt_bboxes:   (B, M, 4)
      gt_classes:  (B, M)    (pad = -1 cho ô trống)

    Returns:
      tp_matrix: (B*N, T) với T=10 (IoU 0.50:0.05:0.95)
    """
    device = pred_bboxes.device
    iou_thresholds = torch.arange(0.5, 1.0, 0.05, device=device)  # (T,)
    T = iou_thresholds.numel()

    # ---- Chuẩn hoá shape ----
    if pred_bboxes.dim() == 2:  # (B*N, 4)
        assert B is not None, "Cần truyền B khi pred đã flatten"
        N = pred_bboxes.shape[0] // B
        pred_bboxes = pred_bboxes.view(B, N, 4)
        pred_conf   = pred_conf.view(B, N)
        pred_classes= pred_classes.view(B, N)
    else:
        B, N, _ = pred_bboxes.shape

    _, M, _ = gt_bboxes.shape

    # ---- Sort theo confidence (desc) ----
    sort_idx = torch.argsort(pred_conf, dim=1, descending=True)                # (B, N)
    pred_bboxes = torch.gather(pred_bboxes, 1, sort_idx[..., None].expand(-1, -1, 4))
    pred_conf   = torch.gather(pred_conf,   1, sort_idx)
    pred_classes= torch.gather(pred_classes,1, sort_idx)

    # ---- Tính IoU: (B, N, M) ----
    ious = bbox_iou_torch(pred_bboxes, gt_bboxes)                              # (B, N, M)

    # ---- Mask GT hợp lệ (nếu có padding -1) ----
    gt_valid = (gt_classes != -1)                                              # (B, M)
    if gt_valid.any():
        # Expand sang (B, 1, M) rồi (B, N, M) khi broadcast
        ious = ious * gt_valid.unsqueeze(1)                                    # zero-out cột GT rỗng

    # ---- Match theo lớp (B, N, M) ----
    cls_match = (pred_classes.unsqueeze(-1) == gt_classes.unsqueeze(1))        # (B, N, M)
    if gt_valid.any():
        cls_match = cls_match & gt_valid.unsqueeze(1)                          # tắt cột rỗng

    # ---- So sánh nhiều IoU thresholds cùng lúc: (B, N, M, T) ----
    meets_thr = ious.unsqueeze(-1) >= iou_thresholds.view(1, 1, 1, T)          # (B, N, M, T)
    matches   = meets_thr & cls_match.unsqueeze(-1)                             # (B, N, M, T)

    # ---- Cưỡng bức one-to-one: giữ True đầu tiên theo thứ tự conf cho mỗi GT, mỗi threshold
    # Với mỗi (B, M, T), dọc theo N (đã sort), lấy True đầu tiên:
    # Ý tưởng: cumsum theo N → chỉ phần tử True đầu tiên có cumsum==1
    m_float   = matches.float()                                                # (B, N, M, T)
    csum      = m_float.cumsum(dim=1)                                          # (B, N, M, T)
    first_true= matches & (csum == 1)                                          # (B, N, M, T) — chỉ giữ True đầu tiên mỗi cột M

    # ---- TP cho từng pred, từng threshold: any theo GT (M) ----
    tp_matrix = first_true.any(dim=2).to(matches.dtype)                        # (B, N, T) -> {0,1}

    # ---- Flatten về (B*N, T) để tương thích compute_ap hiện có ----
    tp_matrix = tp_matrix.reshape(B * N, T)
    return tp_matrix


# %%
import torch
import numpy as np

def compute_ap_torch(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute average precision on GPU tensors.
    tp: torch.Tensor (n_preds, n_iou_thresholds)
    conf: torch.Tensor (n_preds,)
    pred_cls: torch.Tensor (n_preds,)
    target_cls: torch.Tensor (n_targets,)
    Returns: tp, fp, m_pre, m_rec, map50, mean_ap (all as float scalars)
    """
    device = tp.device

    # Sort by confidence
    conf_sort, sort_idx = torch.sort(conf, descending=True)
    tp = tp[sort_idx]
    pred_cls = pred_cls[sort_idx]

    unique_classes, nt = torch.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    n_iou = tp.shape[1]

    ap = torch.zeros((nc, n_iou), device=device)
    p = torch.zeros((nc, 1000), device=device)
    r = torch.zeros((nc, 1000), device=device)
    px = torch.linspace(0, 1, 1000, device=device)

    for ci, c in enumerate(unique_classes):
        mask = pred_cls == c
        nl = nt[ci].item()
        no = mask.sum().item()
        if no == 0 or nl == 0:
            continue

        tpc = tp[mask].cumsum(dim=0)
        fpc = (1 - tp[mask]).cumsum(dim=0)
        recall = tpc / (nl + eps)
        precision = tpc / (tpc + fpc)

        # Interpolation for plotting (optional)
        r[ci] = torch.interp(-px, -conf_sort[mask], recall[:, 0], left=0.0)
        p[ci] = torch.interp(-px, -conf_sort[mask], precision[:, 0], left=1.0)

        # Compute AP for each IoU threshold
        for j in range(n_iou):
            m_rec = torch.cat([torch.tensor([0.0], device=device), recall[:, j], torch.tensor([1.0], device=device)])
            m_pre = torch.cat([torch.tensor([1.0], device=device), precision[:, j], torch.tensor([0.0], device=device)])
            m_pre = torch.flip(torch.maximum.accumulate(torch.flip(m_pre, dims=[0])), dims=[0])
            x = torch.linspace(0, 1, 101, device=device)
            ap[ci, j] = torch.trapz(torch.interp(x, m_rec, m_pre), x)

    # F1 score
    f1 = 2 * p * r / (p + r + eps)
    i = torch.argmax(f1.mean(0))
    p_mean, r_mean, f1_mean = p[:, i], r[:, i], f1[:, i]
    tp_total = (r_mean * nt.to(device)).round()
    fp_total = (tp_total / (p_mean + eps) - tp_total).round()
    ap50, ap_mean = ap[:, 0], ap.mean(1)

    map50, mean_ap = ap50.mean().item(), ap_mean.mean().item()
    return tp_total.cpu().numpy(), fp_total.cpu().numpy(), p_mean.mean().item(), r_mean.mean().item(), map50, mean_ap


# %%
def compute_ap_cpu(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute average precision (AP) on CPU tensors using numpy for interpolation.
    
    Args:
        tp: torch.Tensor, (n_preds, n_iou_thresholds)
        conf: torch.Tensor, (n_preds,)
        pred_cls: torch.Tensor, (n_preds,)
        target_cls: torch.Tensor, (n_targets,)
        eps: float, small number to avoid div by zero
    
    Returns:
        tp_total: np.array, true positives per class
        fp_total: np.array, false positives per class
        m_pre: float, mean precision
        m_rec: float, mean recall
        map50: float, mAP@0.5
        mean_ap: float, mAP@0.5:0.95
    """
    # Move everything to CPU
    tp = tp.cpu().numpy()
    conf = conf.cpu().numpy()
    pred_cls = pred_cls.cpu().numpy()
    target_cls = target_cls.cpu().numpy()

    # Sort by confidence
    sort_idx = np.argsort(-conf)
    tp = tp[sort_idx]
    pred_cls = pred_cls[sort_idx]
    conf = conf[sort_idx]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = len(unique_classes)
    n_iou = tp.shape[1]

    ap = np.zeros((nc, n_iou), dtype=np.float32)
    p = np.zeros((nc, 1000), dtype=np.float32)
    r = np.zeros((nc, 1000), dtype=np.float32)
    px = np.linspace(0, 1, 1000)

    for ci, c in enumerate(unique_classes):
        #mask = pred_cls == c
        mask = (pred_cls.squeeze() == c)
        nl = nt[ci]  # number of labels
        no = mask.sum()  # number of predictions
        if no == 0 or nl == 0:
            continue
        print("mask:",mask.shape)
        tp_class = tp[mask, :]
        if tp_class.shape[0] == 0:
            continue
        tpc = np.cumsum(tp[mask,:], axis=0)
        fpc = np.cumsum(1 - tp[mask], axis=0)
        recall = tpc / (nl + eps)
        precision = tpc / (tpc + fpc)

        conf_masked = conf[mask].ravel()        # flatten về 1D
        recall_masked = recall[:, 0].ravel()
        precision_masked = precision[:, 0].ravel()
        # Interpolation for plotting (numpy)
        # r[ci] = np.interp(-px, -conf_masked, recall_masked, left=0.0)
        # p[ci] = np.interp(-px, -conf_masked, precision_masked, left=1.0)

        
        if len(conf_masked) == 0:  # không có pred cho class này
            r[ci] = 0
            p[ci] = 1
        else:
            r[ci] = np.interp(-px, -conf_masked, recall_masked, left=0.0)
            p[ci] = np.interp(-px, -conf_masked, precision_masked, left=1.0)

        # Compute AP for each IoU threshold
        for j in range(n_iou):
            m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))
            # Precision envelope
            m_pre = np.maximum.accumulate(m_pre[::-1])[::-1]
            x = np.linspace(0, 1, 101)  # 101-point interpolation
            ap[ci, j] = np.trapz(np.interp(x, m_rec, m_pre), x)

    # F1 score
    f1 = 2 * p * r / (p + r + eps)
    i = f1.mean(0).argmax()  # max F1 index
    p_mean = p[:, i]
    r_mean = r[:, i]
    tp_total = np.round(r_mean * nt)
    fp_total = np.round(tp_total / (p_mean + eps) - tp_total)
    ap50 = ap[:, 0]
    ap_mean = ap.mean(1)
    map50 = ap50.mean()
    mean_ap = ap_mean.mean()
    m_pre = p_mean.mean()
    m_rec = r_mean.mean()

    return tp_total, fp_total, m_pre, m_rec, map50, mean_ap

# %%
def compute_ap_cpu(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute average precision (AP) on CPU using numpy.
    
    tp: torch.Tensor (n_preds, n_iou_thresholds)
    conf: torch.Tensor (n_preds,)
    pred_cls: torch.Tensor (n_preds,)
    target_cls: torch.Tensor (n_targets,)
    
    Returns: tp_total, fp_total, m_pre, m_rec, map50, mean_ap
    """
    import numpy as np

    # Move to CPU + numpy
    tp = tp.cpu().numpy()
    conf = conf.cpu().numpy()
    pred_cls = pred_cls.cpu().numpy()
    target_cls = target_cls.cpu().numpy()

    # Sort by confidence descending
    sort_idx = np.argsort(-conf)
    tp = tp[sort_idx]
    pred_cls = pred_cls[sort_idx].squeeze()
    conf = conf[sort_idx]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = len(unique_classes)
    n_iou = tp.shape[1]

    ap = np.zeros((nc, n_iou), dtype=np.float32)
    p = np.zeros((nc, 1000), dtype=np.float32)
    r = np.zeros((nc, 1000), dtype=np.float32)
    px = np.linspace(0, 1, 1000)

    for ci, c in enumerate(unique_classes):
        mask = (pred_cls == c)
        nl = nt[ci]  # number of labels
        no = mask.sum()  # number of predictions
        if no == 0 or nl == 0:
            r[ci] = np.zeros_like(px)
            p[ci] = np.ones_like(px)
            continue

        tp_class = tp[mask, :]  # shape (num_preds_for_class, n_iou)
        tpc = np.cumsum(tp_class, axis=0)
        fpc = np.cumsum(1 - tp_class, axis=0)

        recall = tpc / (nl + eps)
        precision = tpc / (tpc + fpc)

        # Use only first IoU threshold for interp
        conf_masked = conf[mask].ravel()
        recall_masked = recall[:, 0].ravel()
        precision_masked = precision[:, 0].ravel()

        # Ensure same length for interp
        if len(conf_masked) == 0 or len(recall_masked) == 0:
            r[ci] = np.zeros_like(px)
            p[ci] = np.ones_like(px)
        else:
            r[ci] = np.interp(-px, -conf_masked, recall_masked, left=0.0)
            p[ci] = np.interp(-px, -conf_masked, precision_masked, left=1.0)

        # Compute AP for each IoU threshold
        for j in range(n_iou):
            m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))
            m_pre = np.maximum.accumulate(m_pre[::-1])[::-1]  # precision envelope
            x = np.linspace(0, 1, 101)
            ap[ci, j] = np.trapz(np.interp(x, m_rec, m_pre), x)

    # F1 score
    f1 = 2 * p * r / (p + r + eps)
    i = f1.mean(0).argmax()
    p_mean = p[:, i]
    r_mean = r[:, i]

    tp_total = np.round(r_mean * nt)
    fp_total = np.round(tp_total / (p_mean + eps) - tp_total)
    ap50 = ap[:, 0]
    ap_mean = ap.mean(1)
    map50 = ap50.mean()
    mean_ap = ap_mean.mean()
    m_pre = p_mean.mean()
    m_rec = r_mean.mean()

    return tp_total, fp_total, m_pre, m_rec, map50, mean_ap


# %%
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

NUM_CLASSES = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# ===================== Model & Optimizer =====================
model = MyYolo(version='n').to(device)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} million parameters")
print(f"Number of classes (nc): {model.nc}")

criterion = ComputeLoss(model, params)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
num_epochs = 10

# ===================== Training + Validation =====================
for epoch in range(num_epochs):
    # -------- Training --------
    model.train()
    epoch_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train")

    for imgs, targets in train_pbar:
        imgs = imgs.to(device, dtype=torch.float32)
        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

        outputs = model(imgs)
        box_loss, cls_loss, dfl_loss = criterion(outputs, targets)
        loss = box_loss + cls_loss + dfl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        train_pbar.set_postfix({
            "Total": f"{loss.item():.4f}",
            "Cls": f"{cls_loss.item():.4f}",
            "Box": f"{box_loss.item():.4f}",
            "DFL": f"{dfl_loss.item():.4f}"
        })

    print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {epoch_loss/len(train_loader):.4f}")

    # -------- Validation --------
    all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []

    with torch.no_grad():
        # val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val")
        val_pbar = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Val")

        for images, targets in val_pbar:
            images = images.to(device, dtype=torch.float32)
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

            outputs = model(images)
            B = images.shape[0]
            x = torch.cat([i.view(B, model.head.no, -1) for i in outputs], dim=2)
            pred_distri, pred_scores = x.split(split_size=(16*4, NUM_CLASSES), dim=1)
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()

            # Anchors
            anchor_points, stride_tensor = make_anchors(outputs, model.head.stride, offset=0.5)
            anchor_points = anchor_points.to(device)
            pred_bboxes = box_decode_normalize(anchor_points, stride_tensor, pred_distri)
            #pred_bboxes = pred_bboxes.reshape(-1, 4)

            #Predicted class & confidence
            #pred_scores_flat = pred_scores.view(-1, NUM_CLASSES)
            pred_classes = pred_scores.argmax(2)
            pred_conf = pred_scores.max(2).values
            # Match detections (GPU-native)

            gt_bboxes_packed, gt_classes_packed = pack_targets(targets['box'], targets['cls'], targets['idx'], B)
            #tp_matrix = build_tp_matrix_batch(pred_bboxes, pred_conf, pred_classes, targets['box'], targets['cls'], targets['idx'], 32)
            tp_matrix = build_tp_matrix_batch_yolov8_tensor(pred_bboxes, pred_conf, pred_classes,
                                                gt_bboxes_packed, gt_classes_packed, B)
            # Append tensors (still on GPU)
            all_tp.append(tp_matrix)
            all_conf.append(pred_conf.reshape(-1,1))
            all_pred_cls.append(pred_classes.reshape(-1,1))
            all_target_cls.append(targets['cls'])

        # Concatenate all batches (GPU)
        all_tp = torch.cat(all_tp, dim=0)
        all_conf = torch.cat(all_conf, dim=0)
        all_pred_cls = torch.cat(all_pred_cls, dim=0)
        all_target_cls = torch.cat(all_target_cls, dim=0)

        # Compute mAP (GPU)
        tp_arr, fp_arr, m_pre, m_rec, map50, mean_ap = compute_ap_cpu(
            all_tp, all_conf, all_pred_cls, all_target_cls
        )

        # Hiển thị kết quả ngay trên tqdm
        val_pbar.set_postfix({
            "mAP50": f"{map50:.4f}",
            "mAP50-95": f"{mean_ap:.4f}"
        })

    print(f"Validation metrics: mAP50: {map50:.4f}, mAP50-95: {mean_ap:.4f}")

    print(f"Validation metrics: mAP50: {map50:.4f}, mAP50-95: {mean_ap:.4f}")

# %%
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

NUM_CLASSES = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# ===================== Model & Optimizer =====================
model = MyYolo(version='n').to(device)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} million parameters")
print(f"Number of classes (nc): {model.nc}")

criterion = ComputeLoss(model, params)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
num_epochs = 10

# ===================== Training + Validation =====================
for epoch in range(num_epochs):
    # -------- Training --------
    model.train()


    # -------- Validation --------
    all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []

    with torch.no_grad():
        # val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val")
        val_pbar = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Val")

        for images, targets in val_pbar:
            images = images.to(device, dtype=torch.float32)
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
            outputs = model(images)
            B = images.shape[0]
            x = torch.cat([i.view(B, model.head.no, -1) for i in outputs], dim=2)
            pred_distri, pred_scores = x.split(split_size=(16*4, NUM_CLASSES), dim=1)
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()

            # Anchors
            anchor_points, stride_tensor = make_anchors(outputs, model.head.stride, offset=0.5)
            anchor_points = anchor_points.to(device)
            pred_bboxes = box_decode_normalize(anchor_points, stride_tensor, pred_distri)
            #pred_bboxes = pred_bboxes.reshape(-1, 4)

            #Predicted class & confidence
            #pred_scores_flat = pred_scores.view(-1, NUM_CLASSES)
            pred_classes = pred_scores.argmax(2)
            pred_conf = pred_scores.max(2).values
            # Match detections (GPU-native)

            gt_bboxes_packed, gt_classes_packed = pack_targets(targets['box'], targets['cls'], targets['idx'], B)
            #tp_matrix = build_tp_matrix_batch(pred_bboxes, pred_conf, pred_classes, targets['box'], targets['cls'], targets['idx'], 32)
            tp_matrix = build_tp_matrix_batch_yolov8_tensor(pred_bboxes, pred_conf, pred_classes,
                                                gt_bboxes_packed, gt_classes_packed, B)
            # Append tensors (still on GPU)
            all_tp.append(tp_matrix)
            all_conf.append(pred_conf.reshape(-1,1))
            all_pred_cls.append(pred_classes.reshape(-1,1))
            all_target_cls.append(targets['cls'])

        # Concatenate all batches (GPU)
        all_tp = torch.cat(all_tp, dim=0)
        all_conf = torch.cat(all_conf, dim=0)
        all_pred_cls = torch.cat(all_pred_cls, dim=0)
        all_target_cls = torch.cat(all_target_cls, dim=0)
        tp = all_tp.cpu()
        conf = all_conf.cpu()
        pred_cls = all_pred_cls.cpu()
        target_cls = all_target_cls.cpu()
        # Compute mAP (GPU)
        print(tp[0].shape)
        tp_arr, fp_arr, m_pre, m_rec, map50, mean_ap = compute_ap_cpu(
            tp, conf, pred_cls, target_cls
        )

        # Hiển thị kết quả ngay trên tqdm
        val_pbar.set_postfix({
            "mAP50": f"{map50:.4f}",
            "mAP50-95": f"{mean_ap:.4f}"
        })

    print(f"Precision: {m_pre}, Recall: {m_rec}, Validation metrics: mAP50: {map50:.4f}, mAP50-95: {mean_ap:.4f}")


