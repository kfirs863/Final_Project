# EfficiantNet model
# Path: src\models\efficient_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EfficientNet(nn.Module):
