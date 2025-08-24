# %% [markdown]
# # YOLOv8 Custom Training
# 
# ## Các bước thực hiện:
# 1. Setup Google Colab environment
# 2. Import libraries và định nghĩa các components
# 3. Cấu hình dataset tùy chỉnh
# 4. Download và setup Roboflow dataset
# 5. Định nghĩa custom dataset class (không augmentation)
# 6. Training loop với resume capability

# %% [markdown]
# ## 1. Setup Google Colab Environment

# %% [markdown]
# ## 3. Cấu hình Dataset

# %%
# Import các thư viện cần thiết
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset, DataLoader
import yaml
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import shutil
import csv

# Roboflow
try:
    from roboflow import Roboflow
except ImportError:
    os.system("pip install roboflow")
    from roboflow import Roboflow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# ========== CẤU HÌNH CHO DATASET ==========

# Thông tin Roboflow
ROBOFLOW_API_KEY = "YxgjW7IrHbIO7qDyeTKT"
WORKSPACE_NAME = "yolo-tnj0p"
PROJECT_NAME = "wild-animals-detection-yolov8-zw3fg"
VERSION_NUMBER = 1

# Cấu hình model
NUM_CLASSES = 5                         # Số classes trong dataset
INPUT_SIZE = 640                        # Kích thước input image
MODEL_VERSION = 'n'                     # 'n', 's', 'm', 'l', 'x'

# Cấu hình training
BATCH_SIZE = 32                         # Batch size
NUM_EPOCHS = 100                    # Số epochs
LEARNING_RATE = 0.01                    # Learning rate

# Paths cho Google Colab
try:
    DRIVE_SAVE_PATH = "/kaggle/working/"
    os.makedirs(DRIVE_SAVE_PATH, exist_ok=True)

    SAVE_PATH = os.path.join(DRIVE_SAVE_PATH, "custom_yolo_model.pth")
    DATASET_PATH = "/kaggle/input/wild-animals-detection-yolov8"  # Fixed path

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
# ## 4. YOLOv8 Architecture Components
# 
# YOLOv8 bao gồm 3 phần chính:
# - **Backbone**: Trích xuất features từ ảnh đầu vào
# - **Neck**: Kết hợp features ở các scales khác nhau  
# - **Head**: Dự đoán bounding boxes và classifications
# 
# ### 4.1. Basic Building Blocks

# %%
# Conv Block
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Bottleneck
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut

    def forward(self, x):
        x_in = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.shortcut:
            x = x + x_in
        return x

# C2f Block
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks, shortcut=True):
        super().__init__()
        self.mid_channels = out_channels // 2
        self.num_bottlenecks = num_bottlenecks
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.m = nn.ModuleList([Bottleneck(self.mid_channels, self.mid_channels) for _ in range(num_bottlenecks)])
        self.conv2 = Conv((num_bottlenecks + 2) * out_channels // 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x[:, :x.shape[1]//2, :, :], x[:, x.shape[1]//2:, :, :]
        outputs = [x1, x2]
        for i in range(self.num_bottlenecks):
            x1 = self.m[i](x1)
            outputs.insert(0, x1)
        outputs = torch.cat(outputs, dim=1)
        out = self.conv2(outputs)
        return out

# SPPF Block
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv(4 * hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1, ceil_mode=False)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y = torch.cat([x, y1, y2, y3], dim=1)
        y = self.conv2(y)
        return y

# %%
# YOLO Parameters
def yolo_params(version):
    if version == 'n':
        return 1/3, 1/4, 2.0
    elif version == 's':
        return 1/3, 1/2, 2.0
    elif version == 'm':
        return 2/3, 3/4, 1.5
    elif version == 'l':
        return 1.0, 1.0, 1.0
    elif version == 'x':
        return 1.0, 1.25, 1.0

# Backbone
class Backbone(nn.Module):
    def __init__(self, version, in_channels=3):
        super().__init__()
        d, w, r = yolo_params(version)

        # Conv layers
        self.conv_0 = Conv(in_channels, int(64*w), kernel_size=3, stride=2, padding=1)
        self.conv_1 = Conv(int(64*w), int(128*w), kernel_size=3, stride=2, padding=1)
        self.conv_3 = Conv(int(128*w), int(256*w), kernel_size=3, stride=2, padding=1)
        self.conv_5 = Conv(int(256*w), int(512*w), kernel_size=3, stride=2, padding=1)
        self.conv_7 = Conv(int(512*w), int(512*w*r), kernel_size=3, stride=2, padding=1)

        # C2f layers
        self.c2f_2 = C2f(int(128*w), int(128*w), num_bottlenecks=int(3*d), shortcut=True)
        self.c2f_4 = C2f(int(256*w), int(256*w), num_bottlenecks=int(6*d), shortcut=True)
        self.c2f_6 = C2f(int(512*w), int(512*w), num_bottlenecks=int(6*d), shortcut=True)
        self.c2f_8 = C2f(int(512*w*r), int(512*w*r), num_bottlenecks=int(3*d), shortcut=True)

        # SPPF
        self.sppf = SPPF(int(512*w*r), int(512*w*r))

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.c2f_2(x)
        x = self.conv_3(x)
        out1 = self.c2f_4(x)
        x = self.conv_5(out1)
        out2 = self.c2f_6(x)
        x = self.conv_7(out2)
        x = self.c2f_8(x)
        out3 = self.sppf(x)
        return out1, out2, out3

# %% [markdown]
# ### 4.2. Backbone - Feature Extractor
# 
# Backbone sử dụng modified CSPDarknet53 với các components:
# - **Conv layers**: Giảm kích thước và tăng channels
# - **C2f layers**: Cross-stage partial bottleneck
# - **SPPF**: Spatial pyramid pooling fast

# %%
# Upsample
class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

# Neck
class Neck(nn.Module):
    def __init__(self, version):
        super().__init__()
        d, w, r = yolo_params(version)

        self.up = Upsample()
        self.c2f_1 = C2f(in_channels=int(512*w*(1+r)), out_channels=int(512*w), num_bottlenecks=int(3*d), shortcut=False)
        self.c2f_2 = C2f(in_channels=int(768*w), out_channels=int(256*w), num_bottlenecks=int(3*d), shortcut=False)
        self.c2f_3 = C2f(in_channels=int(768*w), out_channels=int(512*w), num_bottlenecks=int(3*d), shortcut=False)
        self.c2f_4 = C2f(in_channels=int(512*w*(1+r)), out_channels=int(512*w*r), num_bottlenecks=int(3*d), shortcut=False)

        self.cv_1 = Conv(in_channels=int(256*w), out_channels=int(256*w), kernel_size=3, stride=2, padding=1)
        self.cv_2 = Conv(in_channels=int(512*w), out_channels=int(512*w), kernel_size=3, stride=2, padding=1)

    def forward(self, x_res_1, x_res_2, x):
        res_1 = x
        x = self.up(x)
        x = torch.cat([x, x_res_2], dim=1)
        res_2 = self.c2f_1(x)
        x = self.up(res_2)
        x = torch.cat([x, x_res_1], dim=1)
        out_1 = self.c2f_2(x)
        x = self.cv_1(out_1)
        x = torch.cat([x, res_2], dim=1)
        out_2 = self.c2f_3(x)
        x = self.cv_2(out_2)
        x = torch.cat([x, res_1], dim=1)
        out_3 = self.c2f_4(x)
        return out_1, out_2, out_3

# %% [markdown]
# ### 4.3. Neck - Feature Pyramid Network
# 
# Neck sử dụng FPN (Feature Pyramid Network) để:
# - **Upsample**: Tăng resolution của features
# - **Feature fusion**: Kết hợp features từ các scales khác nhau
# - **Multi-scale processing**: Xử lý objects ở nhiều kích thước

# %%
# DFL
class DFL(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(1, 2)
        x = x.softmax(1)
        x = self.conv(x)
        return x.view(b, 4, a)

# Head với custom num_classes
class Head(nn.Module):
    def __init__(self, version, ch=16, num_classes=NUM_CLASSES):
        super().__init__()
        self.ch = ch
        self.coordinates = self.ch * 4
        self.nc = num_classes  # Sử dụng custom số classes
        self.no = self.coordinates + self.nc
        self.stride = torch.tensor([8, 16, 32], dtype=torch.float32, device='cuda')

        d, w, r = yolo_params(version=version)

        # Box detection heads
        self.box = nn.ModuleList([
            nn.Sequential(Conv(int(256*w), self.coordinates, kernel_size=3, stride=1, padding=1),
                          Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)),
            nn.Sequential(Conv(int(512*w), self.coordinates, kernel_size=3, stride=1, padding=1),
                          Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)),
            nn.Sequential(Conv(int(512*w*r), self.coordinates, kernel_size=3, stride=1, padding=1),
                          Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1))
        ])

        # Classification heads
        self.cls = nn.ModuleList([
            nn.Sequential(Conv(int(256*w), self.nc, kernel_size=3, stride=1, padding=1),
                          Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)),
            nn.Sequential(Conv(int(512*w), self.nc, kernel_size=3, stride=1, padding=1),
                          Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)),
            nn.Sequential(Conv(int(512*w*r), self.nc, kernel_size=3, stride=1, padding=1),
                          Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1))
        ])

        self.dfl = DFL()

    def forward(self, x):
        for i in range(len(self.box)):
            box = self.box[i](x[i])
            cls = self.cls[i](x[i])
            x[i] = torch.cat((box, cls), dim=1)

        if self.training:
            return x

        anchors, strides = (i.transpose(0, 1) for i in self.make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)
        a, b = self.dfl(box).chunk(2, 1)
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)
        return torch.cat(tensors=(box * strides, cls.sigmoid()), dim=1)

    def make_anchors(self, x, strides, offset=0.5):
        assert x is not None
        anchor_tensor, stride_tensor = [], []
        dtype, device = x[0].dtype, x[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = x[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + offset
            sy = torch.arange(end=h, device=device, dtype=dtype) + offset
            sy, sx = torch.meshgrid(sy, sx)
            anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_tensor), torch.cat(stride_tensor)

# %% [markdown]
# ### 4.4. Head - Detection & Classification
# 
# Head gồm 3 modules chính:
# - **DFL (Distribution Focal Loss)**: Tinh chỉnh bounding box coordinates
# - **Box regression**: Dự đoán vị trí và kích thước objects
# - **Classification**: Dự đoán class probabilities

# %%
# Custom YOLOv8 Model
class CustomYolo(nn.Module):
    def __init__(self, version, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = Backbone(version=version)
        self.neck = Neck(version=version)
        self.head = Head(version=version, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x[0], x[1], x[2])
        return self.head(list(x))

# Test model
model = CustomYolo(version=MODEL_VERSION, num_classes=NUM_CLASSES)

# %% [markdown]
# ## 6. Custom Dataset Class

# %%
class CustomDataset(TorchDataset):
    def __init__(self, image_paths, input_size=INPUT_SIZE):
        self.image_paths = image_paths
        self.input_size = input_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        # print(idx)
        # print(img_path)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load corresponding label file
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')

        # Parse YOLO format labels
        boxes = []
        classes = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])

                        classes.append(class_id)
                        boxes.append([x_center, y_center, width, height])

        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        classes = np.array(classes, dtype=np.int64) if classes else np.zeros((0,), dtype=np.int64)

        # Resize image và adjust boxes
        h, w = image.shape[:2]
        scale = self.input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize image
        image = cv2.resize(image, (new_w, new_h))

        # Create padded image
        padded_image = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded_image[:new_h, :new_w] = image

        # Convert to tensor format
        image_tensor = torch.from_numpy(padded_image.transpose(2, 0, 1)).float() / 255.0

        # Create targets dictionary
        targets = {
            'cls': torch.from_numpy(classes),
            'box': torch.from_numpy(boxes),
            'idx': torch.full((len(classes),), idx, dtype=torch.long)
        }

        return image_tensor, targets

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)

        # Stack images
        images = torch.stack(images, 0)

        # Concatenate targets
        cls_list = []
        box_list = []
        idx_list = []

        for i, target in enumerate(targets):
            if len(target['cls']) > 0:
                cls_list.append(target['cls'])
                box_list.append(target['box'])
                batch_idx = torch.full((len(target['cls']),), i, dtype=torch.long)
                idx_list.append(batch_idx)

        # Concatenate all targets
        all_cls = torch.cat(cls_list, 0) if cls_list else torch.zeros(0, dtype=torch.long)
        all_boxes = torch.cat(box_list, 0) if box_list else torch.zeros(0, 4)
        all_idx = torch.cat(idx_list, 0) if idx_list else torch.zeros(0, dtype=torch.long)

        batch_targets = {
            'cls': all_cls,
            'box': all_boxes,
            'idx': all_idx
        }

        return images, batch_targets

# %% [markdown]
# ## 7. Load Dataset

# %%
def load_dataset_paths(dataset_path):
    train_images = []
    val_images = []

    train_dir = os.path.join(dataset_path, "train", "images")
    if os.path.exists(train_dir):
        for img_file in os.listdir(train_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                train_images.append(os.path.join(train_dir, img_file))

    val_dir = os.path.join(dataset_path, "valid", "images")
    if os.path.exists(val_dir):
        for img_file in os.listdir(val_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                val_images.append(os.path.join(val_dir, img_file))

    return train_images, val_images

# Load dataset
if os.path.exists(DATASET_PATH):
    train_images, val_images = load_dataset_paths(DATASET_PATH)
    
    if len(train_images) > 0 and len(val_images) > 0:
        train_dataset = CustomDataset(train_images, input_size=INPUT_SIZE)
        val_dataset = CustomDataset(val_images, input_size=INPUT_SIZE)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            collate_fn=CustomDataset.collate_fn,
            pin_memory=True if device.type == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            collate_fn=CustomDataset.collate_fn,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        print(f"Dataset loaded: {len(train_images)} train, {len(val_images)} val images")
    else:
        print("No images found. Please check dataset path.")
        train_loader = None
        val_loader = None
else:
    print("Dataset path not found.")
    train_loader = None
    val_loader = None

# %% [markdown]
# ## 9. Training Loop với Resume Capability

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


def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
    
def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """.lp[ki0opk,l]
    boxes: [N,4] dạng (cx, cy, w, h)
    return: [N,4] dạng (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)
def box_decode_normalize(anchor_points, stride_tensor, pred_dist):
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4)
        pred_dist = pred_dist.softmax(3)
        pred_dist = pred_dist.matmul(torch.arange(16, dtype=torch.float, device=device).type(pred_dist.dtype))
        lt, rb = pred_dist.chunk(2, -1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        boxes_decoded = torch.cat(tensors=(x1y1, x2y2), dim=-1)
        # print("Boxes decoded before normalizing:", boxes_decoded[0,:10,:])
        return boxes_decoded*stride_tensor/640 
def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
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
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > confidence_threshold]

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
    def __init__(self, nc=80, top_k=13, alpha=1.0, beta=6.0, eps=1E-9):
        super().__init__()
        self.top_k = top_k
        self.nc = nc
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        batch_size = pd_scores.size(0)
        num_max_boxes = gt_bboxes.size(1)

        if num_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        num_anchors = anc_points.shape[0]
        shape = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        mask_in_gts = torch.cat((anc_points[None] - lt, rb - anc_points[None]), dim=2)
        mask_in_gts = mask_in_gts.view(shape[0], shape[1], num_anchors, -1).amin(3).gt_(self.eps)
        na = pd_bboxes.shape[-2]
        gt_mask = (mask_in_gts * mask_gt).bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([batch_size, num_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([batch_size, num_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, batch_size, num_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=batch_size).view(-1, 1).expand(-1, num_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        bbox_scores[gt_mask] = pd_scores[ind[0], :, ind[1]][gt_mask]  # b, max_num_obj, h*w

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, num_max_boxes, -1, -1)[gt_mask]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[gt_mask]
        overlaps[gt_mask] = compute_iou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        top_k_mask = mask_gt.expand(-1, -1, self.top_k).bool()
        top_k_metrics, top_k_indices = torch.topk(align_metric, self.top_k, dim=-1, largest=True)
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(top_k_indices)
        top_k_indices.masked_fill_(~top_k_mask, 0)

        mask_top_k = torch.zeros(align_metric.shape, dtype=torch.int8, device=top_k_indices.device)
        ones = torch.ones_like(top_k_indices[:, :, :1], dtype=torch.int8, device=top_k_indices.device)
        for k in range(self.top_k):
            mask_top_k.scatter_add_(-1, top_k_indices[:, :, k:k + 1], ones)
        mask_top_k.masked_fill_(mask_top_k > 1, 0)
        mask_top_k = mask_top_k.to(align_metric.dtype)
        mask_pos = mask_top_k * mask_in_gts * mask_gt

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

        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_index]

        # Assigned target scores
        target_labels.clamp_(0)

        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                    dtype=torch.int64,
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
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_box = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        a, b = target_bboxes.chunk(2, -1)
        target = torch.cat((anchor_points - a, b - anchor_points), -1)
        target = target.clamp(0, self.dfl_ch)
        loss_dfl = self.df_loss(pred_dist[fg_mask].view(-1, self.dfl_ch), target[fg_mask])
        loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        return loss_box, loss_dfl

    @staticmethod
    def df_loss(pred_dist, target, reg_max=16):
        """
        pred_dist: (N, reg_max) → logits cho 16 bins
        target: (N, ...) → float, trong [0, reg_max]
        """
        # Clamp target vào [0, reg_max - epsilon]
        eps = 1e-6
        target = target.clamp(0, reg_max - eps)
    
        # left/right bin
        tl = target.floor().long()           # left bin, 0..15
        tr = tl + 1                          # right bin
        tr = tr.clamp(0, reg_max - 1)        # tránh vượt quá 15
    
        # interpolation weights
        wl = tr.float() - target
        wr = 1 - wl
    
        # flatten pred_dist cho cross_entropy
        left_loss = cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape)
        right_loss = cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape)
    
        # nội suy và mean theo chiều cuối
        return (left_loss * wl + right_loss * wr).mean(-1, keepdim=True)


class ComputeLoss:
    def __init__(self, model, params):
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device

        m = model.head  # Head() module

        self.params = params
        self.stride = [8,16,32] 
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.ch
        self.device = device

        self.box_loss = BoxLoss(m.ch).to(device)
        self.cls_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.assigner = Assigner(nc=self.nc, top_k=10, alpha=0.5, beta=6.0)

        self.project = torch.arange(m.ch, dtype=torch.float, device=device)

    def box_decode(self, anchor_points,  pred_dist):
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4)
        pred_dist = pred_dist.softmax(3)
        pred_dist = pred_dist.matmul(self.project.type(pred_dist.dtype))
        lt, rb = pred_dist.chunk(2, -1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        boxes_decoded = torch.cat(tensors=(x1y1, x2y2), dim=-1)
        #print("Boxes decoded before normalizing:", boxes_decoded[0,:10,:])
        return boxes_decoded

    def __call__(self, outputs, targets):
        x = torch.cat([i.view(outputs[0].shape[0], self.no, -1) for i in outputs], dim=2)
        pred_distri, pred_scores = x.split(split_size=(self.reg_max * 4, self.nc), dim=1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        data_type = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        input_size = torch.tensor(outputs[0].shape[2:], device=self.device, dtype=data_type) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(outputs, self.stride, offset=0.5)
        #print("Stride tensor unique:", torch.unique(stride_tensor))
        idx = targets['idx'].view(-1, 1)
        cls = targets['cls'].view(-1, 1)
        box = targets['box']

        targets = torch.cat((idx, cls, box), dim=1).to(self.device)
        if targets.shape[0] == 0:
            gt = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            gt = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets[matches, 1:]
            x = gt[..., 1:5].mul_(input_size[[1, 0, 1, 0]])
            y = torch.empty_like(x)
            dw = x[..., 2] / 2  # half-width
            dh = x[..., 3] / 2  # half-height
            y[..., 0] = x[..., 0] - dw  # top left x
            y[..., 1] = x[..., 1] - dh  # top left y
            y[..., 2] = x[..., 0] + dw  # bottom right x
            y[..., 3] = x[..., 1] + dh  # bottom right y
            gt[..., 1:5] = y
        gt_labels, gt_bboxes = gt.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.box_decode(anchor_points, pred_distri)
        #print("Pred boxes decoded: ", pred_bboxes[0,:10,:])
        assigned_targets = self.assigner(pred_scores.detach().sigmoid(),
                                         (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                                         anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_bboxes, target_scores, fg_mask = assigned_targets
        #print(pred_bboxes[fg_mask][0,:10,:])
        target_scores_sum = max(target_scores.sum(), 1)

        loss_cls = self.cls_loss(pred_scores, target_scores.to(data_type)).sum() / target_scores_sum  # BCE

        # Box loss
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss_box, loss_dfl = self.box_loss(pred_distri,
                                               pred_bboxes,
                                               anchor_points,
                                               target_bboxes,
                                               target_scores,
                                               target_scores_sum, fg_mask)
        
        loss_box *= self.params['box']  # box gain
        loss_cls *= self.params['cls']  # cls gain
        loss_dfl *= self.params['dfl']  # dfl gain
        # Tổng loss
        total_loss = loss_box + loss_cls + loss_dfl

        # Trả về tổng loss (hoặc từng thành phần nếu muốn debug)
        return {"total_loss":total_loss, "box_loss":loss_box, "cls_loss":loss_cls, "dfl_loss":loss_dfl}


# %% [markdown]
# ## 10. Bắt đầu/Resume Training

# %%
import numpy as np

def bbox_iou(box1, box2):
    """
    Compute IoU between two sets of boxes
    box1: (N,4) in xyxy
    box2: (M,4) in xyxy
    return: (N,M) IoU matrix
    """
    N = box1.shape[0]
    M = box2.shape[0]

    # Intersection
    inter_x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    # Union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter_area

    iou = inter_area / (union + 1e-16)
    return iou

import torch

def bbox_iou_torch(boxes1, boxes2):
    """
    boxes1: (B, N, 4) [x1, y1, x2, y2]
    boxes2: (B, M, 4)
    return: (B, N, M)
    """
    # (B, N, 1, 4), (B, 1, M, 4) -> broadcast
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[..., 0:1], boxes1[..., 1:2], boxes1[..., 2:3], boxes1[..., 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[..., 0], boxes2[..., 1], boxes2[..., 2], boxes2[..., 3]

    inter_x1 = torch.max(b1_x1, b2_x1[:, None, :])
    inter_y1 = torch.max(b1_y1, b2_y1[:, None, :])
    inter_x2 = torch.min(b1_x2, b2_x2[:, None, :])
    inter_y2 = torch.min(b1_y2, b2_y2[:, None, :])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union = area1[:, :, None] + area2[:, None, :] - inter_area
    return inter_area / (union + 1e-6)


def build_tp_matrix_batch(pred_bboxes, pred_conf, pred_classes,
                          gt_bboxes, gt_classes):
    """
    Batch version
    pred_bboxes: (B, N, 4)
    pred_conf: (B, N)
    pred_classes: (B, N)
    gt_bboxes: (B, M, 4)
    gt_classes: (B, M)
    
    return: tp_matrix (B, N, T) với T=10 thresholds (0.5:0.05:0.95)
    """
    device = pred_bboxes.device
    iou_thresholds = torch.arange(0.5, 1.0, 0.05, device=device)  # (T,)
    B, N, _ = pred_bboxes.shape
    M = gt_bboxes.shape[1]
    T = len(iou_thresholds)

    # Sort predictions by confidence
    sort_idx = torch.argsort(pred_conf, dim=1, descending=True)
    pred_bboxes = torch.gather(pred_bboxes, 1, sort_idx[..., None].expand(-1, -1, 4))
    pred_classes = torch.gather(pred_classes, 1, sort_idx)

    # IoU (B, N, M)
    ious = bbox_iou_torch(pred_bboxes, gt_bboxes)

    tp_matrix = torch.zeros((B, N, T), device=device)

    for b in range(B):
        assigned_gt = torch.zeros(M, dtype=torch.bool, device=device)
        for n in range(N):
            # Check class match
            cls_matches = (pred_classes[b, n] == gt_classes[b])  # (M,)
            for t, thr in enumerate(iou_thresholds):
                iou_matches = (ious[b, n] >= thr)
                matches = cls_matches & iou_matches & (~assigned_gt)
                if matches.any():
                    gt_idx = torch.argmax(ious[b, n] * matches.float())
                    tp_matrix[b, n, t] = 1
                    assigned_gt[gt_idx] = True
    return tp_matrix


# %%
import torch

def bbox_iou_torch(boxes1, boxes2):
    """
    boxes1: (B, N, 4) [x1, y1, x2, y2]
    boxes2: (B, M, 4)
    return: (B, N, M)
    """
    # (B, N, 1, 4), (B, 1, M, 4) -> broadcast
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[..., 0:1], boxes1[..., 1:2], boxes1[..., 2:3], boxes1[..., 3:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[..., 0], boxes2[..., 1], boxes2[..., 2], boxes2[..., 3]

    inter_x1 = torch.max(b1_x1, b2_x1[:, None, :])
    inter_y1 = torch.max(b1_y1, b2_y1[:, None, :])
    inter_x2 = torch.min(b1_x2, b2_x2[:, None, :])
    inter_y2 = torch.min(b1_y2, b2_y2[:, None, :])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union = area1[:, :, None] + area2[:, None, :] - inter_area
    return inter_area / (union + 1e-6)


def build_tp_matrix_batch(pred_bboxes, pred_conf, pred_classes,
                          gt_bboxes, gt_classes):
    """
    Batch version
    pred_bboxes: (B, N, 4)
    pred_conf: (B, N)
    pred_classes: (B, N)
    gt_bboxes: (B, M, 4)
    gt_classes: (B, M)
    
    return: tp_matrix (B, N, T) với T=10 thresholds (0.5:0.05:0.95)
    """
    device = pred_bboxes.device
    iou_thresholds = torch.arange(0.5, 1.0, 0.05, device=device)  # (T,)
    B, N, _ = pred_bboxes.shape
    M = gt_bboxes.shape[1]
    T = len(iou_thresholds)

    # Sort predictions by confidence
    sort_idx = torch.argsort(pred_conf, dim=1, descending=True)
    pred_bboxes = torch.gather(pred_bboxes, 1, sort_idx[..., None].expand(-1, -1, 4))
    pred_classes = torch.gather(pred_classes, 1, sort_idx)

    # IoU (B, N, M)
    ious = bbox_iou_torch(pred_bboxes, gt_bboxes)

    tp_matrix = torch.zeros((B, N, T), device=device)

    for b in range(B):
        assigned_gt = torch.zeros(M, dtype=torch.bool, device=device)
        for n in range(N):
            # Check class match
            cls_matches = (pred_classes[b, n] == gt_classes[b])  # (M,)
            for t, thr in enumerate(iou_thresholds):
                iou_matches = (ious[b, n] >= thr)
                matches = cls_matches & iou_matches & (~assigned_gt)
                if matches.any():
                    gt_idx = torch.argmax(ious[b, n] * matches.float())
                    tp_matrix[b, n, t] = 1
                    assigned_gt[gt_idx] = True
    return tp_matrix


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
    out_boxes = torch.full((B, M, 4), pad_val, device=gt_classes.device, dtype=torch.float32)
    out_classes = torch.full((B, M), pad_val, device=gt_classes.device, dtype=gt_classes.dtype)

    counter = torch.zeros(B, dtype=torch.long, device=gt_classes.device)

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

    # # Move to CPU + numpy
    # tp = tp.cpu().numpy()
    # # print("tp: ",tp[10:])
    # conf = conf.cpu().numpy()
    # #print("conf before:", conf.shape)
    # pred_cls = pred_cls.cpu().numpy()
    # target_cls = target_cls.cpu().numpy()

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
        recall = recall.squeeze()
        precision = precision.squeeze()

        # Use only first IoU threshold for interp
        conf_masked = conf[mask].ravel()
        #print("conf after:", conf.shape)
        #print("recall first: ", recall.shape)
        recall_masked = np.array(recall)[:, 0].reshape(-1)
        precision_masked = np.array(precision)[:,0].reshape(-1)
        # print("conf: ", conf_masked.shape)
        # print("recall: ", recall_masked.shape)
        # print("precision: ", precision_masked.shape)
        # print("recall type: ", type(recall_masked))
        # print("precision type: ", type(precision_masked))
        # Ensure same length for interp
        if len(conf_masked) == 0 or len(recall_masked) == 0:
            r[ci] = np.zeros_like(px)
            p[ci] = np.ones_like(px)
        else:
            # print("conf: ", conf_masked.shape)
            # print("recall mask: ", recall_masked.shape)
            # print("precision mask: ", precision_masked.shape)
            r[ci] = np.interp(-px, -conf_masked, recall_masked, left=0.0)
            p[ci] = np.interp(-px, -conf_masked, precision_masked, left=1.0)

        # Compute AP for each IoU threshold
        for j in range(n_iou):
            # m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            # m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))
            m_rec = np.concatenate(([0.0], recall[:, j].squeeze(), [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j].squeeze(), [0.0]))
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
def find_latest_checkpoint(checkpoint_dir):
    """
    Tìm checkpoint mới nhất để resume training
    """
    checkpoint_files = []
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.startswith('checkpoint_epoch_') and f.endswith('.pth'):
                epoch_num = int(f.split('_')[2].split('.')[0])
                checkpoint_files.append((epoch_num, os.path.join(checkpoint_dir, f)))

    if checkpoint_files:
        # Sắp xếp theo epoch và lấy checkpoint mới nhất
        checkpoint_files.sort(key=lambda x: x[0])
        return checkpoint_files[-1][1]
    return None

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, checkpoint_dir, is_best=False):
    """
    Save checkpoint để có thể resume training
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'num_classes': NUM_CLASSES,
        'model_version': MODEL_VERSION,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        # Cũng save vào main save path
        torch.save(checkpoint, SAVE_PATH)

    return checkpoint_path

def train_model():
    if train_loader is None:
        print("Error: Dataset not loaded.")
        return

    model = CustomYolo(version=MODEL_VERSION, num_classes=NUM_CLASSES)
    model.to(device)

    criterion = ComputeLoss(model,{"box":7.5, "cls":0.5, "dfl":1.5})
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    train_losses = []
    train_box_losses = []
    train_cls_losses = []
    train_dfl_losses = []
    
    val_losses = []
    val_box_losses = []
    val_cls_losses = []
    val_dfl_losses = []

    precisions = [] 
    recalls = []
    f1s = [] 
    map50s = []
    map5095s = []
    
    lr_history = [] 
    best_val_loss = float('inf')
    start_epoch = 0
    
    history = []
    HISTORY_FILE = os.path.join("/kaggle/working/", "training_history.csv")
    # Định nghĩa các cột giống results.csv của YOLO
    columns = [
        "epoch",
        "train/box_loss", "train/cls_loss", "train/dfl_loss", "train/loss",
        "val/box_loss", "val/cls_loss", "val/dfl_loss", "val/loss"
        "metrics/precision(B)", "metrics/recall(B)", "metrics/f1(B)"
        "metrics/mAP50(B)", "metrics/mAP50-95(B)"
    ]
    
    # Tạo file csv rỗng (chỉ có header)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(HISTORY_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        
        
    print(f"File CSV trống đã tạo: {HISTORY_FILE}")

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_box_loss = 0.0
        train_cls_loss = 0.0
        train_dfl_loss = 0.0 
        
        train_batches = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} - Train')
        
        for batch_idx, (images, targets) in enumerate(train_pbar):
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            losses = criterion(outputs, targets)
            loss = losses["total_loss"] #tensor
            train_box_loss += losses["box_loss"]
            train_cls_loss += losses["cls_loss"]
            train_dfl_loss += losses["dfl_loss"]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Box_loss': f"{losses['box_loss']:.4f}",
                'Cls_loss': f"{losses['cls_loss']:.4f}",
                'Dfl_loss': f"{losses['dfl_loss']:.4f}"
            })

            if device.type == 'cuda':
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    print("Warning: could not clear cache", e)

        # Validation phase
        #model.eval()
        val_loss = 0.0
        val_box_loss = 0.0 
        val_cls_loss = 0.0
        val_dfl_loss = 0.0 
        val_batches = 0
        all_tp = []
        all_conf = []
        all_pred_cls = []
        all_target_cls = [] 
        f1 = tp_arr = fp_arr = m_pre = m_rec = map50 = mean_ap = 0 
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} - Val')
            for images, targets in val_pbar:
                images = images.to(device)
                outputs = model(images)
                losses = criterion(outputs, targets)
                loss = losses['total_loss']
                val_loss += loss.item()
                val_box_loss += losses["box_loss"]
                val_cls_loss += losses["cls_loss"]
                val_dfl_loss += losses["dfl_loss"]
                val_batches += 1
                val_pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Box_loss': f"{losses['box_loss']:.4f}",
                    'Cls_loss': f"{losses['cls_loss']:.4f}",
                    'Dfl_loss': f"{losses['dfl_loss']:.4f}"
                })

                # =================== Decode outputs ===================
                B = images.shape[0]
                x = torch.cat([i.view(B, model.head.no, -1) for i in outputs], dim=2)  # concat 3 scale
                pred_distri, pred_scores = x.split(split_size=(16 * 4, NUM_CLASSES), dim=1)
                pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (B, total_anchors, num_classes)
                pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (B, total_anchors, 4*reg_max)
                
                # Tạo anchor points cho 3 scale
                anchor_points, stride_tensor = make_anchors(outputs, model.head.stride, offset=0.5)
                # Decode bounding boxes
                # print("pred:", pred_bboxes.max())
                pred_bboxes = box_decode_normalize(anchor_points, stride_tensor, pred_distri)  # (B, total_anchors, 4)
                #pred_bboxes = pred_bboxes.view(-1, 4)                 # flatten batch & anchors
                #print("pred:", pred_bboxes[0,500:510,:])
                
                # Predicted class & confidence
                #pred_scores_flat = pred_scores.view(-1, NUM_CLASSES)
                pred_classes = pred_scores.argmax(2)            # (num_pred,)
                pred_conf = pred_scores.max(2).values          # (num_pred,)
                
                # =================== GT ===================
                gt_boxes = targets['box']
                #print("gr_box:", gt_boxes[:10,:])
                gt_classes = targets['cls']
                
                # =================== Match detections ===================
                gt_boxes_xyxy = cxcywh_to_xyxy(targets['box'])
                gt_bboxes_packed, gt_classes_packed = pack_targets(gt_boxes_xyxy, targets['cls'], targets['idx'], B)
                gt_bboxes_tensor = gt_bboxes_packed.to(device=pred_bboxes.device, dtype=pred_bboxes.dtype)
                gt_classes_tensor = gt_classes_packed.to(device=pred_bboxes.device, dtype=pred_bboxes.dtype)
                tp = build_tp_matrix_batch_yolov8_tensor(pred_bboxes,
                                      pred_conf,
                                      pred_classes,
                                      gt_bboxes_tensor, gt_classes_tensor)
                # # Gộp tất cả batch
                # all_tp.append(tp)
                # all_conf.append(pred_conf.cpu().numpy())
                # all_pred_cls.append(pred_classes.cpu().numpy())
                # all_target_cls.append(gt_classes)
            
                all_tp.append(tp.cpu().numpy())
                all_conf.append(pred_conf.reshape(-1,1).cpu().numpy())
                all_pred_cls.append(pred_classes.reshape(-1,1).cpu().numpy())
                all_target_cls.append(targets['cls'].cpu().numpy())
                
                
            # =================== Cuối epoch, tính mAP ===================
            all_tp = np.concatenate(all_tp, axis=0)
            all_conf = np.concatenate(all_conf, axis=0)
            all_pred_cls = np.concatenate(all_pred_cls, axis=0)
            all_target_cls = np.concatenate(all_target_cls, axis=0)
            tp_arr, fp_arr, m_pre, m_rec, map50, mean_ap = compute_ap_cpu(
                    all_tp, all_conf, all_pred_cls, all_target_cls
                )
            print(f"Precision: {m_pre:.4f}, Recall: {m_rec:.4f}, Validation metrics: mAP50: {map50:.10f}, mAP50-95: {mean_ap:.10f}")

        avg_train_loss = train_loss / max(train_batches, 1)
        avg_train_box_loss = train_box_loss / max(train_batches, 1)
        avg_train_cls_loss = train_cls_loss / max(train_batches, 1)
        avg_train_dfl_loss = train_dfl_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        avg_val_box_loss = val_box_loss / max(val_batches, 1)
        avg_val_cls_loss = val_cls_loss / max(val_batches, 1)
        avg_val_dfl_loss = val_dfl_loss / max(val_batches, 1)
        train_losses.append(avg_train_loss)
        train_box_losses.append(avg_train_box_loss)
        train_cls_losses.append(avg_train_cls_loss)
        train_dfl_losses.append(avg_train_dfl_loss)
        val_losses.append(avg_val_loss)
        val_box_losses.append(avg_val_box_loss)
        val_cls_losses.append(avg_val_cls_loss)
        val_dfl_losses.append(avg_val_dfl_loss)
        precisions.append(m_pre)
        recalls.append(m_rec)
        f1s.append(f1)
        map50s.append(map50)
        map5095s.append(mean_ap)
        scheduler.step()
        current_lr = LEARNING_RATE
        lr_history.append(current_lr)
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        # Save checkpoint
        if (epoch + 1) % SAVE_CHECKPOINT_EVERY == 0 or is_best or epoch == NUM_EPOCHS - 1:
            save_checkpoint(model, optimizer, scheduler, epoch, avg_train_loss, avg_val_loss, CHECKPOINT_DIR, is_best)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # history.append({
        #     "epoch": epoch,
        #     "train_box_loss": avg_train_box_loss,
        #     "train_cls_loss": avg_train_cls_loss,
        #     "train_dfl_loss": avg_train_dfl_loss,
        #     "train_loss": avg_train_loss,
        #     "val_box_loss": avg_val_box_loss,
        #     "val_cls_loss": avg_val_cls_loss,
        #     "val_dfl_loss": avg_val_dfl_loss,
        #     "val_loss": avg_val_loss,
        # })
        log_column = [
            f"{avg_train_box_loss:.4f}",
            f"{avg_train_cls_loss:.4f}",
            f"{avg_train_dfl_loss:.4f}",
            f"{avg_train_loss:.4f}",
            f"{avg_val_box_loss:.4f}",
            f"{avg_val_cls_loss:.4f}",
            f"{avg_val_dfl_loss:.4f}",
            f"{avg_val_loss:.4f}",
            f"{m_pre:.4f}",
            f"{m_rec:.4f}",
            f"{f1:.4f}",
            f"{map50:.6f}",
            f"{mean_ap:.6f}",
        ]
        file_exists = os.path.isfile(HISTORY_FILE)
        if file_exists:
            with open(HISTORY_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(log_column)
        
    


    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(start_epoch, start_epoch + len(train_losses)), train_losses, label='Train Loss', color='blue')
    plt.plot(range(start_epoch, start_epoch + len(val_losses)), val_losses, label='Val Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(start_epoch, start_epoch + len(train_losses)), lr_history)
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.grid(True)
    plt.show()

    if len(val_losses) > 1:
        improvement = [(val_losses[0] - val_loss) / val_losses[0] * 100 for val_loss in val_losses]
        plt.plot(range(start_epoch, start_epoch + len(improvement)), improvement)
        plt.title('Validation Loss Improvement (%)')
        plt.xlabel('Epoch')
        plt.ylabel('Improvement (%)')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    return model, train_losses, val_losses

# %%
if train_loader is not None:
    trained_model, train_losses, val_losses = train_model()
else:
    print("Error: Dataset not loaded properly. Please check dataset path and structure.")

# %%
total_params = sum(p.numel() for p in model.parameters())
print(f"Tổng số tham số: {total_params:,}")

# %%
torch.save(model.state_dict(), "/kaggle/working/model_done.pth")

# %% [markdown]
# ## 11. Load Trained Model

# %%
def load_trained_model(model_path):
    if not os.path.exists(model_path):
        return None

    checkpoint = torch.load(model_path, map_location=device)
    model = CustomYolo(
        version=checkpoint.get('model_version', MODEL_VERSION),
        num_classes=checkpoint.get('num_classes', NUM_CLASSES)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

if os.path.exists("/kaggle/input/model-100-epoches/pytorch/default/1/model_done.pth"):
    loaded_model = load_trained_model("/kaggle/input/10-epoch-training/pytorch/default/1/best_model.pth")
else:
    loaded_model = None

# %% [markdown]
# ## 12. Inference Function

# %%
# def inference_single_image(model, image_path, conf_threshold=0.5):
#     if model is None:
#         return None

#     image = cv2.imread(image_path)
#     if image is None:
#         return None

#     original_image = image.copy()
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     h, w = image.shape[:2]
#     scale = INPUT_SIZE / max(h, w)
#     new_h, new_w = int(h * scale), int(w * scale)

#     image = cv2.resize(image, (new_w, new_h))
#     padded_image = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
#     padded_image[:new_h, :new_w] = image

#     image_tensor = torch.from_numpy(padded_image.transpose(2, 0, 1)).float() / 255.0
#     image_tensor = image_tensor.unsqueeze(0).to(device)

#     model.eval()
#     with torch.no_grad():
#         outputs = model(image_tensor)

#     return outputs, original_image


# %%
def xyxy2cxcywh(boxes):
    """
    boxes: Tensor shape (..., 4)  # last dim: x1, y1, x2, y2
    return: Tensor shape (..., 4)  # cx, cy, w, h
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), dim=-1)
import torch
import torchvision
from time import time

def non_max_suppression_with_mask(outputs, confidence_threshold=0.001, iou_threshold=0.7):
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]  # batch size
    nc = outputs.shape[1] - 4  # number of classes
    xc = outputs[:, 4:4 + nc].amax(1) > confidence_threshold  # candidates

    start = time()
    limit = 0.5 + 0.05 * bs
    output = [torch.zeros((0, 6), device=outputs.device)] * bs
    mask_list = [torch.zeros(0, dtype=torch.bool, device=outputs.device)] * bs  # mask

    for index, x in enumerate(outputs):
        x = x.transpose(0, -1)[xc[index]]  # select candidates
        if not x.shape[0]:
            continue

        # boxes + class
        box, cls = x.split((4, nc), 1)
        box = wh2xy(box)  # convert cxcywh -> xyxy

        if nc > 1:
            i, j = (cls > confidence_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > confidence_threshold]

        n = x.shape[0]
        if not n:
            continue

        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        indices = indices[:max_det]

        output[index] = x[indices]

        # tạo mask boolean
        m = torch.zeros(n, dtype=torch.bool, device=x.device)
        m[indices] = True
        mask_list[index] = m

        if (time() - start) > limit:
            break

    return output, mask_list


# %%
# Validation phase
all_tp = []
all_conf = []
all_pred_cls = []
all_target_cls = [] 
criterion = ComputeLoss(model,{"box":7.5, "cls":0.5, "dfl":1.5})
with torch.no_grad():
    val_pbar = tqdm(val_loader, desc=f'Epoch {10}/{NUM_EPOCHS} - Val')
    val_batches = 0 
    for images, targets in val_pbar:
        images = images.to(device)
        model = model.to(device)
        # model.eval()
        with torch.no_grad():
            outputs = model(images)
        val_batches += 1

        # =================== Decode outputs ===================
        B = images.shape[0]
        x = torch.cat([i.view(B, model.head.no, -1) for i in outputs], dim=2)  # concat 3 scale
        pred_distri, pred_scores = x.split(split_size=(16 * 4, NUM_CLASSES), dim=1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (B, total_anchors, num_classes)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (B, total_anchors, 4*reg_max)
        
        # Tạo anchor points cho 3 scale
        anchor_points, stride_tensor = make_anchors(outputs, model.head.stride, offset=0.5)
        # Decode bounding boxes
        # print("pred:", pred_bboxes.max())
        pred_bboxes = box_decode_normalize(anchor_points, stride_tensor, pred_distri)  # (B, total_anchors, 4)
        pred_bboxes_cxcywh = xyxy2cxcywh(pred_bboxes)
        _, mask = non_max_suppression_with_mask(pred_bboxes_cxcywh)
        print(mask[0].sum())
        #pred_bboxes = pred_bboxes.view(-1, 4)                 # flatten batch & anchors
        #print("pred:", pred_bboxes[0,500:510,:])
        
        # Predicted class & confidence
        #pred_scores_flat = pred_scores.view(-1, NUM_CLASSES)
        pred_classes = pred_scores.argmax(2)            # (num_pred,)
        pred_conf = pred_scores.max(2).values          # (num_pred,)
        
        pred_bboxes_kept = [pred_bboxes[mask] for pred_bboxes, mask in zip(pred_bboxes, mask_list)]]
        # =================== GT ===================
        gt_boxes = targets['box']
        #print("gr_box:", gt_boxes[:10,:])
        gt_classes = targets['cls']
        
        # =================== Match detections ===================
        gt_boxes_xyxy = cxcywh_to_xyxy(targets['box'])
        gt_bboxes_packed, gt_classes_packed = pack_targets(gt_boxes_xyxy, targets['cls'], targets['idx'], B)
        gt_bboxes_tensor = gt_bboxes_packed.to(device=pred_bboxes.device, dtype=pred_bboxes.dtype)
        gt_classes_tensor = gt_classes_packed.to(device=pred_bboxes.device, dtype=pred_bboxes.dtype)
        tp = build_tp_matrix_batch_yolov8_tensor(pred_bboxes,
                              pred_conf,
                              pred_classes,
                              gt_bboxes_tensor, gt_classes_tensor)
        print(tp[tp.sum(axis=1) > 0])
        # # Gộp tất cả batch
        # all_tp.append(tp)
        # all_conf.append(pred_conf.cpu().numpy())
        # all_pred_cls.append(pred_classes.cpu().numpy())
        # all_target_cls.append(gt_classes)
        
        all_tp.append(tp.cpu().numpy())
        all_conf.append(pred_conf.reshape(-1,1).cpu().numpy())
        all_pred_cls.append(pred_classes.reshape(-1,1).cpu().numpy())
        all_target_cls.append(targets['cls'].cpu().numpy())
        
        
    # =================== Cuối epoch, tính mAP ===================
    all_tp = np.concatenate(all_tp, axis=0)
    all_conf = np.concatenate(all_conf, axis=0)
    all_pred_cls = np.concatenate(all_pred_cls, axis=0)
    all_target_cls = np.concatenate(all_target_cls, axis=0)
    tp_arr, fp_arr, m_pre, m_rec, map50, mean_ap = compute_ap_cpu(
            all_tp, all_conf, all_pred_cls, all_target_cls
        )
    print(f"Precision: {m_pre:.4f}, Recall: {m_rec:.4f}, Validation metrics: mAP50: {map50:.10f}, mAP50-95: {mean_ap:.10f}")

# %% [markdown]
# ## 13. Summary và Hướng dẫn sử dụng cho Google Colab
# 
# ### 🚀 Tính năng chính của phiên bản Google Colab:
# 
# #### ✅ **Resume Training**
# - Tự động tìm và load checkpoint mới nhất
# - Tiếp tục training từ đúng epoch đã dừng
# - Giữ nguyên optimizer state và learning rate schedule
# - Set `RESUME_TRAINING = True` để kích hoạt
# 
# #### ✅ **Google Colab Optimization**
# - Auto mount Google Drive
# - Optimized memory management
# - GPU detection và configuration
# - Paths tự động cho Colab environment
# 
# #### ✅ **No Data Augmentation**
# - Dataset class đã được simplified
# - Không có horizontal flip hay augmentation khác
# - Chỉ resize và padding cơ bản
# 
# ### 📝 Các bước sử dụng:
# 
# #### **Bước 1: Setup Environment**
# ```python
# # Chạy cell đầu tiên để mount Google Drive
# # Sẽ tự động detect GPU và setup paths
# ```
# 
# #### **Bước 2: Cấu hình Dataset**
# ```python
# ROBOFLOW_API_KEY = "your_actual_api_key"
# WORKSPACE_NAME = "your-workspace"
# PROJECT_NAME = "your-project"
# NUM_CLASSES = 5  # Số classes trong dataset của bạn
# ```
# 
# #### **Bước 3: Download Dataset**
# ```python
# # Uncomment code trong download_roboflow_dataset()
# # Dataset sẽ được download vào /content/roboflow_dataset
# ```
# 
# #### **Bước 4: Start/Resume Training**
# ```python
# # Uncomment trong cell "Bắt đầu Training"
# trained_model, train_losses, val_losses = train_model()
# ```
# 
# ### 🔧 **Resume Training Workflow**
# 
# #### Nếu Colab disconnect trong khi training:
# 1. **Reconnect** và chạy lại từ đầu đến cell cấu hình
# 2. **Set RESUME_TRAINING = True** (mặc định đã True)  
# 3. **Chạy lại train_model()** - sẽ tự động tìm checkpoint mới nhất
# 4. **Training sẽ tiếp tục** từ epoch đã save
# 
# #### Checkpoint Management:
# - **Auto-save** mỗi 10 epochs (configurable)
# - **Best model** được save khi val loss giảm
# - **Checkpoints** lưu trong Google Drive (persistent)
# - **Manual resume** từ epoch cụ thể nếu cần
# 
# ### 💾 **File Structure trên Google Drive**
# ```
# /content/drive/MyDrive/YOLOv8_Training/
# ├── custom_yolo_model.pth          # Best model
# ├── checkpoints/
# │   ├── checkpoint_epoch_10.pth    # Regular checkpoints
# │   ├── checkpoint_epoch_20.pth
# │   ├── best_model.pth             # Copy of best
# │   └── ...
# └── logs/ (optional)
# ```
# 
# ### ⚡ **Performance Tips cho Colab**
# 
# #### Memory Management:
# - **Batch size**: Start với 16, giảm xuống 8 nếu OOM
# - **num_workers**: Set 2 cho dataloaders  
# - **pin_memory**: True khi dùng GPU
# - **torch.cuda.empty_cache()**: Auto cleanup sau mỗi epoch
# 
# #### Training Interruption:
# - **Google Colab timeout**: ~12 hours max
# - **Save frequency**: Mỗi 10 epochs (có thể adjust)
# - **Best model**: Always saved khi có improvement
# - **Resume**: Seamless với RESUME_TRAINING=True
# 
# ### 🐛 **Troubleshooting**
# 
# #### Common Issues:
# - **Out of Memory**: Giảm BATCH_SIZE xuống 8 hoặc 4
# - **Dataset not found**: Check DATASET_PATH và folder structure  
# - **Resume không work**: Check CHECKPOINT_DIR có checkpoints không
# - **Slow training**: Verify GPU được sử dụng (`device = cuda`)
# 
# #### Debug Commands:
# ```python
# # Check GPU status
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# 
# # Check checkpoint
# latest = find_latest_checkpoint(CHECKPOINT_DIR)
# print(f"Latest checkpoint: {latest}")
# 
# # Check dataset
# print(f"Train images: {len(train_images)}")
# print(f"Val images: {len(val_images)}")
# ```
# 
# ### 🎯 **Training Time Estimates**
# Với dataset 12,000 samples, 5 classes:
# - **Google Colab GPU (T4)**: ~4-6 giờ cho 100 epochs  
# - **Google Colab CPU**: ~50-70 giờ (không khuyến nghị)
# - **Colab Pro GPU (A100/V100)**: ~2-3 giờ cho 100 epochs
# 
# ### 📊 **Monitoring Training**
# - Progress bars hiển thị realtime loss
# - Training curves plot sau khi hoàn thành  
# - Checkpoint saves được log ra console
# - Best model được highlight khi tìm thấy
# 
# ---
# ### 🚦 **Ready to Train!**
# Tất cả đã setup xong. Uncomment dòng training trong cell 10 để bắt đầu!


