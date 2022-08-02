import torch as t
from torch import nn, optim
from dataset import make_loader
import os
from model import FINet
from loss import iteration_for_loss
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


