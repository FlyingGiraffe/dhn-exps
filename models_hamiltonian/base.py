import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


class BaseHamiltonianNet(nn.Module):
  
  def __init__(self, model_config, dtype=torch.float32):
    super(BaseHamiltonianNet, self).__init__()
    self.dtype = dtype
  
  def get_condition(self, data):
    def unsqueeze_1_dim(x):
      if len(x.shape) == 1:
        x = x[..., None]
      return x

    batch_size = data['q'].shape[0]
    device = data['q'].device
    cond = torch.empty((batch_size, 0))
    for k in data['cond_dict']:
      c = data['cond_dict'][k].reshape(batch_size, -1)
      c = unsqueeze_1_dim(c)
      cond = torch.cat([cond, c], dim=-1)
    cond = cond.to(self.dtype).to(device)
    return cond
  
  def get_latent_code(self, data):
    return self.codebook(data['idx'])
  
  def get_input_coords(self, data, return_metadata=False):
    '''overwrite this function to add data pre-processing
    '''
    q, p = data['q'], data['L_grad_dq']
    q = q * 2 / np.pi
    p_scale = torch.abs(p).max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    p = p / p_scale
    
    if return_metadata:
      metadata = {'p_scale': p_scale[:, 0, 0]}
      return q, p, metadata
    else:
      return q, p
  
  def postprocess_output_coords(self, q, p):
    '''overwrite this function to add data post-processing
    '''
    return q, p
  
  def normalize_time(self, t):
    t = (t - self.t_span[0]) / (self.t_span[1] - self.t_span[0])
    return t
  
  def get_losses(self, data, loss_config):
    raise NotImplementedError
  
  def plot_traj_image(self, q_dict, t):
    t = t.detach().cpu().numpy()

    colors = ['#6A5B6E', '#E6B89C', '#EAD2AC', '#4281A4', '#9CAFB7']

    fig, ax = plt.subplots()
    for i, k in enumerate(q_dict):
      q = q_dict[k].squeeze(-1).detach().cpu().numpy()
      ax.plot(t, q, linewidth=3, color=colors[i], label=k)
    
    ax.set_xlabel('time')
    ax.set_ylabel('q')
    ax.legend()

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_tensor = torch.tensor(image).permute(2, 0, 1)

    plt.close()
    return image_tensor
  
  def get_traj_image_vis(self, q_dict, t, num_vis=None):
    num_vis = t.shape[0] if num_vis is None else num_vis
    image_tensor_list = [
      self.plot_traj_image({k: q_dict[k][i] for k in q_dict}, t[i])
      for i in range(min(num_vis, t.shape[0]))
    ]
    image_tensor = torch.stack(image_tensor_list, axis=0)
    return image_tensor
  
  def get_vis_dict(self, dict_vals, num_vis=None):
    raise NotImplementedError
  
  def gen_sequence(self, *args, **kwargs):
    raise NotImplementedError
  
  def inference(self, data):
    raise NotImplementedError

  def gen_results_for_eval(self, data, gen_config):
    raise NotImplementedError
  
  def extract_get_losses(self, data, loss_config):
    raise NotImplementedError
  
  def extract_get_vis_dict(self, dict_vals, num_vis=None):
    raise NotImplementedError
  
  def extract_inference(self, data):
    raise NotImplementedError