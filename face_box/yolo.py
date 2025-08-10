import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
from torchvision.ops import generalized_box_iou_loss
 
# loading YOLO Model

def load_model():
    model = YOLO("model/best.pt")
    return model