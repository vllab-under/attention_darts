import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model):
    self.network_momentum = 0.9
    self.network_weight_decay = 3e-4
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=6e-4, betas=(0.5, 0.999), weight_decay=1e-3)

  def step(self, loss_search):
    self.optimizer.zero_grad()
    self._backward_step(loss_search)
    self.optimizer.step()

  def _backward_step(self, loss_search):
    loss_search.backward()
