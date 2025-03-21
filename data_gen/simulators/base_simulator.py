import os
import numpy as np


class BaseSimulator(object):

  def __init__(self, config):

    self.save_dir = config.save_dir
    self.vis_dir = os.path.join(config.save_dir, 'vis')

    self.t_span = list(config.t_span)
    self.num_timesteps = config.num_timesteps

    self.cond_dict = {}
    self.constant_counter = 0

    self.color_palette = ['#6A5B6E', '#E6B89C', '#EAD2AC', '#9CAFB7', '#4281A4']
  
  def init_state(self):
    raise NotImplementedError
  
  def lagrangian(self, q, dq):
    raise NotImplementedError

  def lagrangian_grad_q(self, q, dq):
    raise NotImplementedError

  def lagrangian_grad_dq(self, q, dq):
    raise NotImplementedError
  
  def sample_trajectory(self):
    raise NotImplementedError
  
  def visualize(self, traj, i_data):
    return
  
  def load_constant(self, c, name=None):
    if isinstance(c, list) or isinstance(c, tuple):
      c = np.random.uniform(c[0], c[1])
      if name is None:
        name = f'cond{self.constant_counter}'
        self.constant_counter += 1
      self.cond_dict[name] = c
    return c