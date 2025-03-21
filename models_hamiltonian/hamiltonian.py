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
    self.num_stages = len(model_config.stage_step_size)
    self.stage_step_size = model_config.stage_step_size
    self.stage_block_size = model_config.stage_block_size
    self.stage_block_step = model_config.stage_block_step
    self.num_noise_scales = model_config.num_noise_scales
    self.state_base_with_noise = model_config.state_base_with_noise
    
    self.codebook = Codebook(model_config.codebook)

    self.stage_H_plus, self.stage_H_minus = nn.ModuleList([]), nn.ModuleList([])
    for i_stage in range(self.num_stages):
      block_size = model_config.stage_block_size[i_stage]
      model_config.network.block_size = block_size

      self.stage_H_plus.append(get_model(model_config.network))
      self.stage_H_minus.append(get_model(model_config.network))
    
  def forward_H_plus(self, q, p, q_mask, p_mask, z, i_stage):
    return self.stage_H_plus[i_stage](q, p, q_mask, p_mask, z).squeeze(-1)

  def forward_H_minus(self, p, q, p_mask, q_mask, z, i_stage):
    return self.stage_H_minus[i_stage](q, p, q_mask, p_mask, z).squeeze(-1)
  
  def get_block_feature(self, feat, block_size):
    '''
      input: feat, shape [B, T, C]
      output: feat_block, shape [B, T-S+1, S, C], S is block_size
    '''
    batch_size, num_steps, c_dim = feat.shape
    idx_base = torch.arange(0, block_size, dtype=torch.int64)
    idx_range = torch.arange(0, num_steps - block_size + 1, dtype=torch.int64)
    idx = idx_base[None] + idx_range[:, None]
    feat_block = feat[:, idx]
    return feat_block
  
  def compute_H_plus_grads(self, q, p, q_mask, p_mask, z, i_stage):
    forward_fn = partial(self.forward_H_plus, i_stage=i_stage)
    func_grad = torch.vmap(torch.func.jacrev(forward_fn, argnums=(0, 1)))
    D1_H, D2_H = func_grad(q, p, q_mask, p_mask, z)
    return D1_H, D2_H
  
  def compute_H_minus_grads(self, p, q, p_mask, q_mask, z, i_stage):
    forward_fn = partial(self.forward_H_minus, i_stage=i_stage)
    func_grad = torch.vmap(torch.func.jacrev(forward_fn, argnums=(0, 1)))
    D1_H, D2_H = func_grad(p, q, p_mask, q_mask, z)
    return D1_H, D2_H

  def compute_H_plus_state_updates(
    self, q, p_next, q_mask, p_next_mask, z, i_stage,
    q_base=None, p_base=None
  ):
    '''
      Discrete right Hamiltonian.
    '''
    q_base = q if q_base is None else q_base
    p_base = p_next if p_base is None else p_base
    D1_H, D2_H = self.compute_H_plus_grads(q, p_next, q_mask, p_next_mask, z, i_stage)
    q_next_pred = q_base + D2_H
    p_pred = p_base + D1_H
    return self.postprocess_output_coords(q_next_pred, p_pred)
  
  def compute_H_minus_state_updates(
    self, q_next, p, q_next_mask, p_mask, z, i_stage,
    q_base=None, p_base=None,
  ):
    '''
      Discrete left Hamiltonian.
    '''
    q_base = q_next if q_base is None else q_base
    p_base = p if p_base is None else p_base
    # note here p goes first
    D1_H, D2_H = self.compute_H_minus_grads(p, q_next, p_mask, q_next_mask, z, i_stage)
    q_pred = q_base - D1_H
    p_next_pred = p_base - D2_H
    return self.postprocess_output_coords(q_pred, p_next_pred)
  
  def add_rand_noise(self, feat, mask, noise_scale=None):
    # map zeros entries in mask to random a in [0, 1], the input mask should be binary
    if noise_scale is None:
      a = torch.rand_like(mask)
    else:
      a = torch.ones_like(mask) * noise_scale
    mask = mask + a * (1 - mask)

    noise = torch.randn_like(feat)
    feat = feat * mask + noise * (1 - mask)
    return feat, mask
  
  def compute_state_denoise(
    self, q_in, p_in, q_out, p_out, z, state_update_fn,
    q_in_mask=None, p_in_mask=None, q_out_mask=None, p_out_mask=None,
    add_noise=True, noise_scale=None,
  ):
    q_in_mask = torch.ones_like(q_in) if q_in_mask is None else q_in_mask
    p_in_mask = torch.ones_like(p_in) if p_in_mask is None else p_in_mask
    q_out_mask = torch.ones_like(q_out) if q_out_mask is None else q_out_mask
    p_out_mask = torch.ones_like(p_out) if p_out_mask is None else p_out_mask

    if add_noise:
      q_in_noised, q_in_mask = self.add_rand_noise(q_in, q_in_mask, noise_scale=noise_scale)
      p_in_noised, p_in_mask = self.add_rand_noise(p_in, p_in_mask, noise_scale=noise_scale)
    else:
      q_in_noised = q_in
      p_in_noised = p_in
    
    q_base = q_in_noised if self.state_base_with_noise else q_in
    p_base = p_in_noised if self.state_base_with_noise else p_in
    
    q_pred, p_pred = state_update_fn(
      q_in_noised, p_in_noised, q_in_mask, p_in_mask, z,
      q_base=q_base, p_base=p_base,
    )

    q_pred = q_out * q_out_mask + q_pred * (1 - q_out_mask)
    p_pred = p_out * p_out_mask + p_pred * (1 - p_out_mask)

    return self.postprocess_output_coords(q_pred, p_pred)
  
  def compute_block_updates(
    self,
    q_block_in, p_block_in, z, state_update_fn, block_size,
  ):
    batch_size, num_blocks, _, q_dim = q_block_in.shape
    z_dim = z.shape[-1]

    q_block_pred, p_block_pred = state_update_fn(
      q_block_in.reshape(-1, block_size, q_dim),
      p_block_in.reshape(-1, block_size, q_dim),
      torch.ones_like(q_block_in).reshape(-1, block_size, q_dim),
      torch.ones_like(p_block_in).reshape(-1, block_size, q_dim),
      z[:, None].repeat(1, num_blocks, 1).reshape(-1, z_dim),
    )
    q_block_pred = q_block_pred.reshape(batch_size, -1, block_size, q_dim)
    p_block_pred = p_block_pred.reshape(batch_size, -1, block_size, q_dim)

    return q_block_pred, p_block_pred
  
  def compute_block_denoise(
    self,
    q_block_in, p_block_in, q_block_out, p_block_out, z, state_update_fn, block_size,
    q_in_mask, p_in_mask, q_out_mask=None, p_out_mask=None,
    add_noise=True, noise_scale=None,
  ):
    batch_size, num_blocks, _, q_dim = q_block_in.shape
    z_dim = z.shape[-1]

    q_in_mask = q_in_mask.reshape(-1, block_size, q_dim)
    p_in_mask = p_in_mask.reshape(-1, block_size, q_dim)
    q_out_mask = q_out_mask.reshape(-1, block_size, q_dim) if q_out_mask is not None else None
    p_out_mask = p_out_mask.reshape(-1, block_size, q_dim) if p_out_mask is not None else None

    q_block_denoise, p_block_denoise = self.compute_state_denoise(
      q_block_in.reshape(-1, block_size, q_dim),
      p_block_in.reshape(-1, block_size, q_dim),
      q_block_out.reshape(-1, block_size, q_dim),
      p_block_out.reshape(-1, block_size, q_dim),
      z[:, None].repeat(1, num_blocks, 1).reshape(-1, z_dim),
      state_update_fn=state_update_fn,
      q_in_mask=q_in_mask,
      p_in_mask=p_in_mask,
      q_out_mask=q_out_mask,
      p_out_mask=p_out_mask,
      add_noise=add_noise,
      noise_scale=noise_scale,
    )
    q_block_denoise = q_block_denoise.reshape(batch_size, -1, block_size, q_dim)
    p_block_denoise = p_block_denoise.reshape(batch_size, -1, block_size, q_dim)

    return q_block_denoise, p_block_denoise
  
  def get_train_stage_inputs(self, q, p, i_stage, crop_interval=None):
    q = q[:, ::self.stage_step_size[i_stage]]
    p = p[:, ::self.stage_step_size[i_stage]]
    if crop_interval[1] > 0:
      q = q[:, crop_interval[0]:crop_interval[1]]
      p = p[:, crop_interval[0]:crop_interval[1]]
    return q, p
  
  def get_train_stage_blocks(self, q, p, i_stage):
    block_size = self.stage_block_size[i_stage]
    block_step = self.stage_block_step[i_stage]
    
    q_block = self.get_block_feature(q, block_size=block_size)  # [B, T-S+1, S, q_dim]
    p_block = self.get_block_feature(p, block_size=block_size)  # [B, T-S+1, S, q_dim]
    num_blocks = q_block.shape[1]

    q_src = q_block[:, :num_blocks - block_step]
    p_src = p_block[:, :num_blocks - block_step]
    q_tgt = q_block[:, block_step:]
    p_tgt = p_block[:, block_step:]

    return q_src, p_src, q_tgt, p_tgt
  
  def get_train_stage_masks(self, q_block, i_stage):
    raise NotImplementedError
  
  def get_stage_losses(self, q, p, z, i_stage, loss_config):
    q, p = self.get_train_stage_inputs(q, p, i_stage, crop_interval=loss_config.crop_interval)
    q_src, p_src, q_tgt, p_tgt = self.get_train_stage_blocks(q, p, i_stage)

    H_plus_update_fn = partial(self.compute_H_plus_state_updates, i_stage=i_stage)
    H_minus_update_fn = partial(self.compute_H_minus_state_updates, i_stage=i_stage)

    q_tgt_pred, p_src_pred = self.compute_block_updates(
      q_src, p_tgt, z,
      state_update_fn=H_plus_update_fn,
      block_size=self.stage_block_size[i_stage],
    )
    q_src_pred, p_tgt_pred = self.compute_block_updates(
      q_tgt, p_src, z,
      state_update_fn=H_minus_update_fn,
      block_size=self.stage_block_size[i_stage],
    )

    q_src_mask, p_src_mask, q_tgt_mask, p_tgt_mask = self.get_train_stage_masks(q_src, i_stage)

    q_tgt_denoise, p_src_denoise = self.compute_block_denoise(
      q_src, p_tgt, q_tgt, p_src, z,
      state_update_fn=H_plus_update_fn,
      block_size=self.stage_block_size[i_stage],
      q_in_mask=q_src_mask, p_in_mask=p_tgt_mask,
      q_out_mask=q_tgt_mask if loss_config.use_out_mask else torch.zeros_like(q_tgt_mask),
      p_out_mask=p_src_mask if loss_config.use_out_mask else torch.zeros_like(p_src_mask),
    )
    q_src_denoise, p_tgt_denoise = self.compute_block_denoise(
      q_tgt, p_src, q_src, p_tgt, z,
      state_update_fn=H_minus_update_fn,
      block_size=self.stage_block_size[i_stage],
      q_in_mask=q_tgt_mask, p_in_mask=p_src_mask,
      q_out_mask=q_src_mask if loss_config.use_out_mask else torch.zeros_like(q_src_mask),
      p_out_mask=p_tgt_mask if loss_config.use_out_mask else torch.zeros_like(p_tgt_mask),
    )

    loss_eom_plus = F.mse_loss(q_tgt_pred, q_tgt) + F.mse_loss(p_src_pred, p_src)
    loss_eom_minus = F.mse_loss(q_src_pred, q_src) + F.mse_loss(p_tgt_pred, p_tgt)
    loss_eom = loss_eom_plus + loss_eom_minus

    loss_denoise_plus = F.mse_loss(q_tgt_denoise, q_tgt) + F.mse_loss(p_src_denoise, p_src)
    loss_denoise_minus = F.mse_loss(q_src_denoise, q_src) + F.mse_loss(p_tgt_denoise, p_tgt)
    loss_denoise = loss_denoise_plus + loss_denoise_minus

    loss_train = (
      loss_eom * loss_config.weight_eom
      + loss_denoise * loss_config.weight_denoise
    )

    dict_losses = {
      f'loss_train/stage_{i_stage}': loss_train.item(),
      f'loss_train_eom/stage_{i_stage}': loss_eom.item(),
      f'loss_train_denoise/stage_{i_stage}': loss_denoise.item(),
    }
    return loss_train, dict_losses
  
  def get_losses(self, data, loss_config):
    q, p = self.get_input_coords(data)
    z = self.get_latent_code(data)

    loss_train = 0.0
    dict_losses = {}
    for i_stage in range(self.num_stages):
      stage_loss_train, stage_dict_losses = self.get_stage_losses(q, p, z, i_stage, loss_config)
      loss_train = loss_train + stage_loss_train
      dict_losses = dict_losses | stage_dict_losses
    
    dict_losses['loss_train/train'] = loss_train

    return loss_train, dict_losses