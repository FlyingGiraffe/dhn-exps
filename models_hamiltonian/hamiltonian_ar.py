from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

from models_hamiltonian.hamiltonian import HamiltonianNet


class HamiltonianNet(HamiltonianNet):
  
  def get_train_stage_masks(self, q_block, i_stage):
    block_size = self.stage_block_size[i_stage]
    block_step = self.stage_block_step[i_stage]

    q_src_mask = torch.ones_like(q_block)
    p_tgt_mask = torch.ones_like(q_block)
    p_tgt_mask[:, :, block_size - block_step:] = 0

    q_tgt_mask = torch.ones_like(q_block)
    p_src_mask = torch.ones_like(q_block)
    q_tgt_mask[:, :, block_size - block_step:] = 0

    return q_src_mask, p_src_mask, q_tgt_mask, p_tgt_mask
  
  def get_gen_init(self, q, p, num_init_steps=None):
    q_pred = torch.zeros_like(q, dtype=self.dtype)
    p_pred = torch.zeros_like(p, dtype=self.dtype)
    if num_init_steps is None:
      # minimal initial steps needed
      q_pred[:, :self.stage_block_size[0]] = q[:, :self.stage_block_size[0]]
      p_pred[:, :self.stage_block_size[0]] = p[:, :self.stage_block_size[0]]
    else:
      # for fair comparison with other experimental setups
      q_pred[:, :num_init_steps] = q[:, :num_init_steps]
      p_pred[:, :num_init_steps] = p[:, :num_init_steps]
    state_mask = torch.zeros_like(q)
    state_mask[:, :self.stage_block_size[0]] = 1
    return q_pred, p_pred, state_mask, num_init_steps
  
  def get_gen_gt(self, q, p, t):
    num_steps = t.shape[1]
    q_gt = q[:, :num_steps - 1:self.stage_step_size[-1]]
    p_gt = p[:, :num_steps - 1:self.stage_step_size[-1]]
    t = t[:, :num_steps - 1:self.stage_step_size[-1]]
    return q_gt, p_gt, t
  
  def gen_sequence(
    self, q, p, z, num_init_steps=None, num_denoise_steps=1, update_step_size=None
  ):
    q_pred, p_pred, state_mask, num_init_steps = self.get_gen_init(q, p, num_init_steps)
    batch_size, num_steps, q_dim = q.shape    

    i_stage = 0
    block_size = self.stage_block_size[i_stage]
    block_step = self.stage_block_step[i_stage]
    step_size = self.stage_step_size[i_stage]

    idx_src = torch.arange(0, block_size, dtype=torch.int64)
    idx_tgt = idx_src + block_step

    for i_step in range(num_steps - 1):
      if idx_tgt[-1] >= num_steps:
        break

      q_src = q_pred[:, None, idx_src]
      p_src = p_pred[:, None, idx_src]
      q_tgt_pred = q_pred[:, None, idx_tgt]
      p_tgt_pred = p_pred[:, None, idx_tgt]

      H_plus_update_fn = partial(self.compute_H_plus_state_updates, i_stage=i_stage)
      H_minus_update_fn = partial(self.compute_H_minus_state_updates, i_stage=i_stage)

      q_src_mask = torch.ones_like(q_src)
      p_src_mask = torch.ones_like(p_src)
      q_tgt_mask = torch.ones_like(q_src)
      p_tgt_mask = torch.ones_like(p_src)
      q_tgt_mask[:, :, block_size - block_step:] = 0
      p_tgt_mask[:, :, block_size - block_step:] = 0

      for i_denoise in range(num_denoise_steps + 1):
        noise_scale = i_denoise / num_denoise_steps
        q_tgt_pred, _ = self.compute_block_denoise(
          q_src, p_tgt_pred, q_tgt_pred, p_src, z,
          state_update_fn=H_plus_update_fn,
          block_size=self.stage_block_size[i_stage],
          q_in_mask=q_src_mask, p_in_mask=p_tgt_mask,
          q_out_mask=q_tgt_mask, p_out_mask=p_src_mask,
          add_noise=True,
          noise_scale=noise_scale,
        )
        _, p_tgt_pred = self.compute_block_denoise(
          q_tgt_pred, p_src, q_src, p_tgt_pred, z,
          state_update_fn=H_minus_update_fn,
          block_size=self.stage_block_size[i_stage],
          q_in_mask=q_tgt_mask, p_in_mask=p_src_mask,
          q_out_mask=q_src_mask, p_out_mask=p_tgt_mask,
          add_noise=True,
          noise_scale=noise_scale,
        )
      
      update_step_size = block_step if update_step_size is None else update_step_size
      
      idx_seq = torch.arange(1, update_step_size + 1, dtype=torch.int64) + idx_src[-1]
      idx_pred = torch.arange(0, update_step_size, dtype=torch.int64) + block_size - block_step
      q_pred[:, idx_seq] = q_tgt_pred[:, 0, idx_pred]
      p_pred[:, idx_seq] = p_tgt_pred[:, 0, idx_pred]
      idx_src += update_step_size
      idx_tgt += update_step_size

      if num_init_steps is not None:
        q_pred[:, :num_init_steps] = q[:, :num_init_steps]
        p_pred[:, :num_init_steps] = p[:, :num_init_steps]
    
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
  
  def gen_results_for_eval(self, data, gen_config):
    num_init_steps = gen_config.num_init_steps
    num_denoise_steps = gen_config.num_denoise_steps
    update_step_size = gen_config.update_step_size

    q, p, metadata = self.get_input_coords(data, return_metadata=True)
    t = self.normalize_time(data['time'])
    z = self.get_latent_code(data)
    q_gt, p_gt, t = self.get_gen_gt(q, p, t)

    q_pred, p_pred = self.gen_sequence(
      q_gt, p_gt, z,
      num_init_steps=num_init_steps,
      num_denoise_steps=num_denoise_steps,
      update_step_size=update_step_size
    )

    dict_results = {
      't': t.detach().cpu().numpy(),
      'q_gt': q_gt.detach().cpu().numpy(), 'p_gt': p_gt.detach().cpu().numpy(),
      'q_pred': q_pred.detach().cpu().numpy(), 'p_pred': p_pred.detach().cpu().numpy(),
      'p_scale': metadata['p_scale'].detach().cpu().numpy(),
    }

    return dict_results
  
  def extract_get_losses(self, data, loss_config):
    loss_train, dict_losses = super(HamiltonianNet, self).get_losses(data, loss_config)
    return loss_train, dict_losses
  
  def extract_get_vis_dict(self, dict_vals, num_vis=None):
    dict_vis = self.get_vis_dict(dict_vals, num_vis=num_vis)
    return dict_vis
  
  def extract_inference(self, data):
    dict_losses, dict_vals = self.inference(data)
    return dict_losses, dict_vals