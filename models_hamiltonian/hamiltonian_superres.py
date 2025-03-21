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

    self.train_step_span = model_config.train_step_span
  
  def get_train_stage_masks(self, q_block, i_stage):
    q_src_mask = torch.zeros_like(q_block)
    p_tgt_mask = torch.zeros_like(q_block)
    q_src_mask[:, :, 0] = 1
    p_tgt_mask[:, :, 1] = 1

    q_tgt_mask = torch.zeros_like(q_block)
    p_src_mask = torch.zeros_like(q_block)
    q_tgt_mask[:, :, 1] = 1
    p_src_mask[:, :, 0] = 1

    return q_src_mask, p_src_mask, q_tgt_mask, p_tgt_mask
  
  def get_train_stage_inputs(self, q, p, i_stage, crop_interval=None):
    idx_start = self.train_step_span[0]
    idx_end = self.train_step_span[1]
    q = q[:, idx_start:idx_end]
    p = p[:, idx_start:idx_end]
    
    q = q[:, ::self.stage_step_size[i_stage]]
    p = p[:, ::self.stage_step_size[i_stage]]
    return q, p
  
  def get_gen_init(self, q, p):
    step_size_0 = 2 * self.stage_step_size[0] // self.stage_step_size[-1]

    q_pred = torch.zeros_like(q, dtype=self.dtype)
    p_pred = torch.zeros_like(p, dtype=self.dtype)
    q_pred[:, ::step_size_0] = q[:, ::step_size_0]
    p_pred[:, ::step_size_0] = p[:, ::step_size_0]
    state_mask = torch.zeros_like(q)
    state_mask[:, ::step_size_0] = 1
    return q_pred, p_pred, state_mask
  
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
  
  def gen_sequence(self, q, p, z, num_denoise_steps=1):
    q_pred, p_pred, state_mask = self.get_gen_init(q, p)

    for i_stage in range(self.num_stages):

      block_size = self.stage_block_size[i_stage]
      step_size = self.stage_step_size[i_stage] // self.stage_step_size[-1]
      step_size_prev = step_size * 2

      q_block = self.get_block_feature(q_pred[:, ::step_size], block_size=block_size)
      p_block = self.get_block_feature(p_pred[:, ::step_size], block_size=block_size)
      mask_block = self.get_block_feature(state_mask[:, ::step_size], block_size=block_size)

      q_src_pred = q_block[:, ::2]
      p_src_pred = p_block[:, ::2]
      q_tgt_pred = q_block[:, 1::2]
      p_tgt_pred = p_block[:, 1::2]

      H_plus_update_fn = partial(self.compute_H_plus_state_updates, i_stage=i_stage)
      H_minus_update_fn = partial(self.compute_H_minus_state_updates, i_stage=i_stage)

      q_src_mask = mask_block[:, ::2]
      p_src_mask = mask_block[:, ::2]
      q_tgt_mask = mask_block[:, 1::2]
      p_tgt_mask = mask_block[:, 1::2]

      for i_denoise in range(num_denoise_steps + 1):
        noise_scale = i_denoise / num_denoise_steps
        q_tgt_pred, p_src_pred = self.compute_block_denoise(
          q_src_pred, p_tgt_pred, q_tgt_pred, p_src_pred, z,
          state_update_fn=H_plus_update_fn,
          block_size=self.stage_block_size[i_stage],
          q_in_mask=q_src_mask, p_in_mask=p_tgt_mask,
          q_out_mask=q_tgt_mask, p_out_mask=p_src_mask,
          add_noise=True,
          noise_scale=noise_scale,
        )
        q_src_pred, p_tgt_pred = self.compute_block_denoise(
          q_tgt_pred, p_src_pred, q_src_pred, p_tgt_pred, z,
          state_update_fn=H_minus_update_fn,
          block_size=self.stage_block_size[i_stage],
          q_in_mask=q_tgt_mask, p_in_mask=p_src_mask,
          q_out_mask=q_src_mask, p_out_mask=p_tgt_mask,
          add_noise=True,
          noise_scale=noise_scale,
        )
        q_src_pred[:, :, 1] = q_tgt_pred[:, :, 0] = (q_src_pred[:, :, 1] + q_tgt_pred[:, :, 0]) * 0.5
        p_tgt_pred[:, :, 0] = p_src_pred[:, :, 1] = (p_tgt_pred[:, :, 0] + p_src_pred[:, :, 1]) * 0.5
      
      q_pred[:, step_size::step_size_prev] = q_src_pred[:, :, 1]
      p_pred[:, step_size::step_size_prev] = p_tgt_pred[:, :, 0]
      state_mask[:, step_size::step_size_prev] = 1
    
    return q_pred, p_pred
  
  def get_vis_dict(self, dict_vals, num_vis=None):
    t = dict_vals['t']
    q_gt = dict_vals['q_gt']
    p_gt = dict_vals['p_gt']

    num_denoise_steps_list = [1, self.num_noise_scales]

    q_vis_dict, p_vis_dict = {}, {}
    q_vis_dict['q gt'] = q_gt[..., 0]
    p_vis_dict['p gt'] = p_gt[..., 0]
    for num_denoise_steps in num_denoise_steps_list:
      q_vis_dict[f'q pred (denoise steps =  {num_denoise_steps})'] = dict_vals[f'q_pred_{num_denoise_steps}'][..., 0]
      p_vis_dict[f'p pred (denoise steps = {num_denoise_steps})'] = dict_vals[f'p_pred_{num_denoise_steps}'][..., 0]
    traj_q_vis = self.get_traj_image_vis(q_vis_dict, t, num_vis=num_vis)
    traj_p_vis = self.get_traj_image_vis(p_vis_dict, t, num_vis=num_vis)

    dict_vis = {
      'traj_q': traj_q_vis,
      'traj_p': traj_p_vis,
    }
    
    return dict_vis
  
  def inference(self, data):
    q, p = self.get_input_coords(data)
    t = self.normalize_time(data['time'])
    z = self.get_latent_code(data)
    q_gt, p_gt, t = self.get_gen_gt(q, p, t)

    num_denoise_steps_list = [1, self.num_noise_scales]

    q_pred_list, p_pred_list = {}, {}
    for num_denoise_steps in num_denoise_steps_list:
      q_pred_list[str(num_denoise_steps)], p_pred_list[str(num_denoise_steps)] = self.gen_sequence(
        q_gt, p_gt, z, num_denoise_steps=num_denoise_steps)
    
    dict_losses = {}
    for num_denoise_steps in num_denoise_steps_list:
      dict_losses[f'loss_eval/q_{num_denoise_steps}'] = F.mse_loss(q_pred_list[str(num_denoise_steps)], q_gt).item()

    dict_vals = {
      't': t,
      'q_gt': q_gt,
      'p_gt': p_gt,
    }

    for num_denoise_steps in num_denoise_steps_list:
      dict_vals[f'q_pred_{num_denoise_steps}'] = q_pred_list[str(num_denoise_steps)]
      dict_vals[f'p_pred_{num_denoise_steps}'] = p_pred_list[str(num_denoise_steps)]

    return dict_losses, dict_vals
  
  def extract_get_losses(self, data, loss_config):
    q, p = self.get_input_coords(data)
    z = self.get_latent_code(data)
    loss_train, dict_losses = self.get_stage_losses(q, p, z, 0, loss_config)
    return loss_train, dict_losses
  
  def extract_get_vis_dict(self, dict_vals, num_vis=None):
    dict_vis = self.get_vis_dict(dict_vals, num_vis=num_vis)
    return dict_vis
  
  def extract_inference(self, data):
    dict_losses, dict_vals = self.inference(data)
    return dict_losses, dict_vals