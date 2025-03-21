from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

from models_hamiltonian.hamiltonian import HamiltonianNet


class HamiltonianNet(HamiltonianNet):

  def __init__(self, model_config, dtype=torch.float32):
    super(HamiltonianNet, self).__init__(model_config, dtype=dtype)

    self.z_dim = model_config.embedding_dim
    self.c_dim = model_config.c_dim
    self.lin_probe = nn.Linear(self.z_dim, self.c_dim)
  
  def get_train_stage_masks(self, q_block, i_stage):
    batch_size, num_blocks, block_size, q_dim = q_block.shape

    idx_src = torch.randperm(block_size)[:block_size//2]
    idx_tgt = torch.randperm(block_size)[:block_size//2]

    q_src_mask = torch.ones_like(q_block)
    p_tgt_mask = torch.ones_like(q_block)
    q_src_mask[:, :, idx_src] = 0
    p_tgt_mask[:, :, idx_tgt] = 0

    q_tgt_mask = torch.ones_like(q_block)
    p_src_mask = torch.ones_like(q_block)
    q_tgt_mask[:, :, idx_tgt] = 0
    p_src_mask[:, :, idx_src] = 0

    return q_src_mask, p_src_mask, q_tgt_mask, p_tgt_mask
  
  def get_losses(self, data, loss_config):
    loss_train, dict_losses = super(HamiltonianNet, self).get_losses(data, loss_config)

    z = self.get_latent_code(data)
    z = z.detach()
    cond_pred = self.lin_probe(z)
    cond = self.get_condition(data)

    loss_lin_probe = F.mse_loss(cond_pred, cond)
    loss_train = loss_train + loss_lin_probe
    dict_losses['loss_train/lin_probe'] = loss_lin_probe.item()

    return loss_train, dict_losses
  
  def get_vis_dict(self, dict_vals, num_vis=None):
    return {}
  
  def inference(self, data):
    dict_losses = {}
    dict_vals = {}
    return dict_losses, dict_vals
  
  def extract_get_losses(self, data, loss_config):
    loss_train, dict_losses = super(HamiltonianNet, self).get_losses(data, loss_config)
    return loss_train, dict_losses
  
  def extract_get_vis_dict(self, dict_vals, num_vis=None):
    return {}
  
  def extract_inference(self, data):
    z = self.get_latent_code(data)
    z = z.detach()
    cond_pred = self.lin_probe(z)
    cond = self.get_condition(data)
    loss_lin_probe = F.mse_loss(cond_pred, cond)

    dict_losses = {
      'loss_eval/lin_probe': loss_lin_probe.item(),
    }
    dict_vals = {}
    return dict_losses, dict_vals

