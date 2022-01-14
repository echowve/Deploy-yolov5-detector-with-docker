import argparse
import time
from pathlib import Path

import cv2
from numpy.core.defchararray import mod
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.plots import colors, plot_one_box
from yolov5.utils.torch_utils import select_device
import cv2
import numpy as np


@torch.no_grad()
def detect(model,

           imgsz=640,  # inference size (pixels)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=50,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           half=False,  # use FP16 half-precision inference
           im0=None
           ):

    
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    
    stride = int(torch.max(model.stride)) # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size


    if half:
        model.half()  # to FP16

    img = letterbox(im0, imgsz, stride)[0]
    img = np.transpose(img, (2, 0, 1))

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference

    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process detections
    for det in pred:  # detections per image
    
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        return det.detach().cpu().numpy().tolist()


if __name__=='__main__':
    check_requirements(exclude=('tensorboard', 'thop'))

    source = 'test.jpg'
    img0 = cv2.imread(source)  # BGR
    imgsz=640
    show_name_list = ['person']
    device = select_device('cpu')
    model = attempt_load('yolov5s.pt', map_location=device)  # load FP32 model
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    names = model.module.names if hasattr(model, 'module') else model.names

    print('model_loaded')


    t1 = time.time()
    detections = detect(model=model, device=device, im0=img0, imgsz=imgsz)
    t2 = time.time()

    gn = np.array(img0.shape)[[1, 0, 1, 0]]

    for de in detections:
        xyxy= de[:4]
        conf, cls = de[4:]
        c = int(cls)
        if names[c] not in show_name_list: continue
        xywh = (xyxy2xywh(xyxy.view(1, 4)) / gn).view(-1).tolist()
        label = f'{names[c]} {conf:.2f}'
        plot_one_box(xyxy, img0, label=label, color=colors(c, True), line_thickness=1)

    cv2.imshow('result', img0)
    cv2.waitKey(0)
    print('time pass %.2fs'%(t2 -t1))
