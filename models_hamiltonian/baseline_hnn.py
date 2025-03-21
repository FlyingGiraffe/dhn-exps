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
    self.step_size = model_config.step_size
    
    self.codebook = Codebook(model_config.codebook)
    self.H = get_model(model_config.network)
    
  def forward_H(self, q, p, z):
    return self.H(q, p, z).squeeze(-1)
  
  def compute_H_grads(self, q, p, z):
    func_grad = torch.vmap(torch.func.jacrev(self.forward_H, argnums=(0, 1)))
    D1_H, D2_H = func_grad(q, p, z)
    return D1_H, D2_H

  def compute_H_state_updates(self, q, p, z):
    D1_H, D2_H = self.compute_H_grads(q, p, z)
    q_next_pred = q + D2_H
    p_next_pred = p - D1_H
    return self.postprocess_output_coords(q_next_pred, p_next_pred)
  
  def compute_H_state_updates_leapfrog(self, q, p, z):
    D1_H, _ = self.compute_H_grads(q, p, z)
    p_half = p - D1_H * 0.5

    _, D2_H_half = self.compute_H_grads(q, p_half, z)
    q_next = q + D2_H_half * 0.5

    D1_H_next, _ = self.compute_H_grads(q_next, p_half, z)
    p_next = p_half - D1_H_next * 0.5
    return self.postprocess_output_coords(q_next, p_next)
  
  def get_train_inputs(self, q, p, loss_config):
    q = q[:, ::self.step_size]
    p = p[:, ::self.step_size]
    if loss_config.crop_interval[1] > 0:
      q = q[:, loss_config.crop_interval[0]:loss_config.crop_interval[1]]
      p = p[:, loss_config.crop_interval[0]:loss_config.crop_interval[1]]
    return q, p
  
  def get_losses(self, data, loss_config):
    q, p = self.get_input_coords(data)
    z = self.get_latent_code(data)

    q, p = self.get_train_inputs(q, p, loss_config)
    batch_size, num_steps, q_dim = q.shape
    z_dim = z.shape[-1]

    q_src, p_src = q[:, :num_steps-1], p[:, :num_steps-1]
    q_tgt, p_tgt = q[:, 1:], p[:, 1:]

    q_tgt_pred, p_tgt_pred = self.compute_H_state_updates(
      q_src.reshape(-1, q_dim),
      p_src.reshape(-1, q_dim),
      z[:, None].repeat(1, num_steps - 1, 1).reshape(-1, z_dim),
    )
    q_tgt_pred = q_tgt_pred.reshape(batch_size, num_steps - 1, q_dim)
    p_tgt_pred = p_tgt_pred.reshape(batch_size, num_steps - 1, q_dim)

    loss_eom = F.mse_loss(q_tgt_pred, q_tgt) + F.mse_loss(p_tgt_pred, p_tgt)
    loss_train = loss_eom
    
    dict_losses = {
      f'loss_train/train': loss_train.item(),
      f'loss_train/eom': loss_eom.item(),
    }

    return loss_train, dict_losses