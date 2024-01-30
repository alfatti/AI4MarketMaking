import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
from torch.nn import Module, Linear, BatchNorm1d, Tanh
from numba import cuda
from torch import autograd
import torch.nn as nn
import torch.n


class NuarlBox(nn.Module):
  """
  Base subnetwork that introduced in DGM.

  Args:
  indim (int): input dimension
  outdim (int) : number of node in hidden layer

  """
  def __init__(self,
              indim=100,
              outdim = 50
              ):
    super().__init__()
    self.acitivision = nn.Tanh()
    self.z = nn.Linear(outdim,outdim,bias=False)
    self.g = nn.Linear(outdim,outdim,bias=False)
    self.r = nn.Linear(outdim,outdim,bias=False)
    self.h = nn.Linear(outdim,outdim,bias=False)
    self.z1 = nn.Linear(outdim,outdim)
    self.g1 = nn.Linear(outdim,outdim)
    self.r1 = nn.Linear(outdim,outdim)
    self.h1 = nn.Linear(outdim,outdim)
    self.s1 = nn.Linear(indim,outdim)
    self.s2 = nn.Linear(indim ,outdim)
    self.s3 = nn.Linear(indim ,outdim)
    self.s4 = nn.Linear(indim ,outdim)







  def forward(self, x,s):

    z1 =self.z(s) + self.s1(x)
    z1 = self.acitivision(z1)
    g1 = self.g(s) + self.s2(x)
    g1 = self.acitivision(g1)
    r1 = self.r(s) + self.s3(x)
    r1 = self.acitivision(r1)
    h1 = self.h(torch.mul(s,r1)) + self.s4(x)
    h1 = self.acitivision(h1)
    s2 = torch.mul((1.0-g1),h1) + torch.mul(z1,s)
    return s2


class DGM(nn.Module):
  """
    Base Model

    Args:
    dim (int): input dimension
    layersize (int) : number of node in hidden layer

  """
  def __init__(self,

              dim=1,
              layersize = 10,
              ):
    super().__init__()


    self.dim = dim
    self.layer1 = nn.Linear(self.dim  , layersize)
    self.module1 = NuarlBox(indim = self.dim ,outdim = layersize)
    self.module2 = NuarlBox(indim = self.dim ,outdim = layersize)
    self.module3 = NuarlBox(indim = self.dim,outdim = layersize)

    self.last_layer = nn.Linear(layersize,1)

    self.acitivision  = nn.Tanh()


  def forward(self,y):

    s1=self.layer1(y)
    s1= self.acitivision(s1)
    s2 = self.module1(y,s1)
    s3 = self.module2(y,s2)
    s4 = self.module3(y,s3)
    out =self.last_layer(s4)
    return out
