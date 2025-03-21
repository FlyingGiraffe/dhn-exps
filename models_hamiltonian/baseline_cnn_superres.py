from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

from models_hamiltonian.base import BaseHamiltonianNet
from models import get_model
from models.codebook import Codebook


class HamiltonianNet(BaseHamiltonianNet):
  
  def __init__(self, model_config, dtype=torch.float32):
    super(HamiltonianNet, self).__init__(model_config, dtype=dtype)

    self.t_span = model_config.t_span
    self.stage_step_size = model_config.stage_step_size
    self.train_step_span = model_config.train_step_span
    
    self.network = get_model(model_config.network)
    
  def forward(self, q, p):
    return self.network(q, p)
  
  def get_train_inputs(self, q, p, i_stage):
    idx_start = self.train_step_span[0]
    idx_end = self.train_step_span[1]
    q = q[:, idx_start:idx_end]
    p = p[:, idx_start:idx_end]

    q = q[:, ::self.stage_step_size[i_stage]]
    p = p[:, ::self.stage_step_size[i_stage]]
    return q, p
  
  def get_losses(self, data, loss_config):
    q, p = self.get_input_coords(data)

    q_pred, p_pred = self.get_train_inputs(q, p, 0)
    q_pred, p_pred = self.forward(q_pred, p_pred)

    q_tgt, p_tgt = self.get_train_inputs(q, p, 1)
    loss_train = F.mse_loss(q_pred, q_tgt) + F.mse_loss(p_pred, p_tgt)

    dict_losses = {
      'loss_train/train': loss_train.item(),
    }
    return loss_train, dict_losses

  def get_gen_init(self, q, p):
    step_size_0 = self.stage_step_size[0] // self.stage_step_size[-1]
    q_pred = q[:, ::step_size_0]
    p_pred = p[:, ::step_size_0]
    return q_pred, p_pred
  
  def get_gen_gt(self, q, p, t):
    idx_start = self.train_step_span[0]
    idx_end = self.train_step_span[1]
    q = q[:, idx_start:idx_end]
    p = p[:, idx_start:idx_end]
    t = t[:, idx_start:idx_end]

    q_gt = q[:, ::self.stage_step_size[-1]]
    p_gt = p[:, ::self.stage_step_size[-1]]
    t = t[:, ::self.stage_step_size[-1]]
    return q_gt, p_gt, t
  
  def gen_sequence(self, q, p):
    q_pred, p_pred = self.get_gen_init(q, p)
    q_pred, p_pred = self.forward(q_pred, p_pred)
    return q_pred, p_pred
  
  def get_vis_dict(self, dict_vals, num_vis=None):
    t = dict_vals['t']
    q_gt = dict_vals['q_gt']
    p_gt = dict_vals['p_gt']
    q_pred = dict_vals['q_pred']
    p_pred = dict_vals['p_pred']

    traj_q_vis = self.get_traj_image_vis(
      {'q gt':  q_gt[..., 0], 'q pred': q_pred[..., 0]}, t, num_vis=num_vis)
    traj_p_vis = self.get_traj_image_vis(
      {'p gt':  p_gt[..., 0], 'p pred': p_pred[..., 0]}, t, num_vis=num_vis)

    dict_vis = {
      'traj_q': traj_q_vis,
      'traj_p': traj_p_vis,
    }
    return dict_vis
  
  def inference(self, data):
    q, p = self.get_input_coords(data)
    t = self.normalize_time(data['time'])
    q_gt, p_gt, t = self.get_gen_gt(q, p, t)

    q_pred, p_pred = self.gen_sequence(q_gt, p_gt)
    loss_eval_q = F.mse_loss(q_pred, q_gt).item()

    dict_losses = {
      'loss_eval/q': loss_eval_q,
    }
    dict_vals = { 't': t,
      'q_gt': q_gt, 'p_gt': p_gt,
      'q_pred': q_pred, 'p_pred': p_pred,
    }

    return dict_losses, dict_vals
  
  def extract_get_losses(self, data, loss_config):
    return torch.tensor(0.0, requires_grad=True), {}
  
  def extract_get_vis_dict(self, dict_vals, num_vis=None):
    dict_vis = self.get_vis_dict(dict_vals, num_vis=num_vis)
    return dict_vis
  
  def extract_inference(self, data):
    dict_losses, dict_vals = self.inference(data)
    return dict_losses, dict_vals