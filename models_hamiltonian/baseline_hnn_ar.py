from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

from models_hamiltonian.baseline_hnn import HamiltonianNet


class HamiltonianNet(HamiltonianNet):
  
  def get_gen_init(self, q, p, num_init_steps=None):
    q_pred = torch.zeros_like(q, dtype=self.dtype)
    p_pred = torch.zeros_like(p, dtype=self.dtype)
    if num_init_steps is None:
      # minimal initial steps needed
      q_pred[:, 0] = q[:, 0]
      p_pred[:, 0] = p[:, 0]
    else:
      # for fair comparison with other experimental setups
      q_pred[:, :num_init_steps] = q[:, :num_init_steps]
      p_pred[:, :num_init_steps] = p[:, :num_init_steps]
    return q_pred, p_pred, num_init_steps
  
  def get_gen_gt(self, q, p, t):
    num_steps = t.shape[1]
    q_gt = q[:, :num_steps - 1:self.step_size]
    p_gt = p[:, :num_steps - 1:self.step_size]
    t = t[:, :num_steps - 1:self.step_size]
    return q_gt, p_gt, t
  
  def gen_sequence(self, q, p, z, num_init_steps=None, fwd_algorithm='forward_euler'):
    q_pred, p_pred, num_init_steps = self.get_gen_init(q, p, num_init_steps)
    batch_size, num_steps, q_dim = q.shape

    for i_step in range(num_steps - 1):
      q_src = q_pred[:, i_step]
      p_src = p_pred[:, i_step]
      if fwd_algorithm == 'forward_euler':
        q_tgt_pred, p_tgt_pred = self.compute_H_state_updates(q_src, p_src, z)
      elif fwd_algorithm == 'leapfrog':
        q_tgt_pred, p_tgt_pred = self.compute_H_state_updates_leapfrog(q_src, p_src, z)
      else:
        raise NotImplementedError('Forward algorithm not implemented.')
      q_pred[:, i_step + 1] = q_tgt_pred
      p_pred[:, i_step + 1] = p_tgt_pred
      
      if num_init_steps is not None:
        q_pred[:, :num_init_steps] = q[:, :num_init_steps]
        p_pred[:, :num_init_steps] = p[:, :num_init_steps]
    
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
    z = self.get_latent_code(data)
    q_gt, p_gt, t = self.get_gen_gt(q, p, t)

    q_pred, p_pred = self.gen_sequence(q_gt, p_gt, z, fwd_algorithm='forward_euler')
    loss_eval_q = F.mse_loss(q_pred, q_gt).item()

    dict_losses = {
      'loss_eval/q': loss_eval_q,
    }
    dict_vals = { 't': t,
      'q_gt': q_gt, 'p_gt': p_gt,
      'q_pred': q_pred, 'p_pred': p_pred,
    }

    return dict_losses, dict_vals
  
  def gen_results_for_eval(self, data, gen_config):
    num_init_steps = gen_config.num_init_steps

    q, p, metadata = self.get_input_coords(data, return_metadata=True)
    t = self.normalize_time(data['time'])
    z = self.get_latent_code(data)
    q_gt, p_gt, t = self.get_gen_gt(q, p, t)

    q_pred, p_pred = self.gen_sequence(
      q_gt, p_gt, z,
      num_init_steps=num_init_steps,
      fwd_algorithm=gen_config.fwd_algorithm,
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