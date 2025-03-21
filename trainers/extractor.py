"""Freeze network, optimize for the latent code.
"""

import os
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from input_pipeline import create_dataloader
from models_hamiltonian import get_model_hamiltoinian


def get_sub_static_dict(model_state_dict, prefix=''):
  sub_state_dict = {
    k.replace(prefix, ''): v
    for k, v in model_state_dict.items() if k.startswith(prefix)
  }
  return sub_state_dict


def get_filtered_static_dict(model_state_dict, prefix_list=['']):
  def is_filter_key(k):
    for prefix in prefix_list:
      if k.startswith(prefix):
        return True
    return False

  filtered_state_dict = {
    k: v for k, v in model_state_dict.items() 
    if not is_filter_key(k)
  }
  return filtered_state_dict


class Extractor(object):

  def __init__(self, config, extract_dir='extract', dtype=torch.float32):

    self.config = config
    self.dtype = dtype
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # general
    self.workdir = self.config.workdir
    self.extract_dir = os.path.join(self.workdir, extract_dir)
    # optim
    self.num_epochs = self.config.optim.num_epochs
    self.lr = self.config.optim.lr
    # logging
    self.per_save_epochs = self.config.logging.per_save_epochs
    self.per_save_tmp_epochs = self.config.logging.per_save_tmp_epochs
    self.per_eval_epochs = self.config.logging.per_eval_epochs
    self.num_eval_batches = self.config.logging.num_eval_batches
    self.num_vis = self.config.logging.num_vis

    # dataloaders
    self.train_loader = create_dataloader(config.data, split='train', shuffle=False)
    self.test_loader = create_dataloader(config.data, split='test', shuffle=False)

    # networks
    self.hamiltonian_net = get_model_hamiltoinian(config.model, dtype=dtype)
    self.hamiltonian_net = self.hamiltonian_net.to(self.dtype).to(self.device)
    self.trainable_prefix_list = ['codebook'] #['codebook', 'lin_probe']
    self.load_pretrained_checkpoint()

    # optimizer
    self.optimizer = optim.Adam(self.hamiltonian_net.parameters(), lr=self.lr, weight_decay=1e-4)
    self.scheduler = CosineAnnealingLR(self.optimizer, self.num_epochs, eta_min=self.lr)
  
  def preprocess_data(self, data):
    for k in data:
      if isinstance(data[k], torch.Tensor) and k != 'idx':
        data[k] = data[k].to(self.dtype).to(self.device)
    data['idx'] = data['idx'].to(torch.long).to(self.device)
    return data
  
  def load_pretrained_checkpoint(self):
    ckpt_files = glob.glob(os.path.join(self.workdir, 'checkpoint_ep*.pth'))
    
    if not ckpt_files:
      print('No checkpoint files found in the directory.')
      return 0

    ckpt_files.sort(key=lambda f: int(f.split('_ep')[-1].split('.pth')[0]))
    latest_ckpt_path = ckpt_files[-1]

    checkpoint = torch.load(latest_ckpt_path)
    filtered_state_dict = get_filtered_static_dict(
      checkpoint['model_state_dict'], prefix_list=self.trainable_prefix_list)
    self.hamiltonian_net.load_state_dict(filtered_state_dict, strict=False)

    for name, param in self.hamiltonian_net.named_parameters():
      if name in filtered_state_dict:
          param.requires_grad = False
  
  def train_step(self):
    for data in self.test_loader:
      self.optimizer.zero_grad()
      data = self.preprocess_data(data)
      loss_train, dict_losses = self.hamiltonian_net.extract_get_losses(data, self.config.loss)
      loss_train.backward()
      self.optimizer.step()
    self.scheduler.step()
    return dict_losses
  
  def eval_step(self):
    self.hamiltonian_net.eval()
    with torch.no_grad():
      dict_losses_all = []
      for i, data in enumerate(self.test_loader):
        if i >= self.num_eval_batches:
          break
        data = self.preprocess_data(data)
        dict_losses, dict_vals = self.hamiltonian_net.extract_inference(data)
        dict_losses_all.append(dict_losses)
        if i == 0:
          dict_vis = self.hamiltonian_net.extract_get_vis_dict(dict_vals, num_vis=self.num_vis)
      dict_losses_mean = {}
      for k in dict_losses_all[0]:
        dict_losses_mean[k] = np.mean([dict_losses[k] for dict_losses in dict_losses_all])
      return dict_losses_mean, dict_vis
  
  def train_and_eval(self):
    writer = SummaryWriter(self.extract_dir)

    for epoch in range(self.num_epochs + 1):
      self.hamiltonian_net.train()
      dict_losses = self.train_step()
      for k in dict_losses:
        writer.add_scalar(k, dict_losses[k], epoch)
      
      if epoch % self.per_eval_epochs == 0:
        dict_losses_eval, dict_vis_eval = self.eval_step()
        for k in dict_losses_eval:
          writer.add_scalar(k, dict_losses_eval[k], epoch)
        for k in dict_vis_eval:
          image_tensor = dict_vis_eval[k]
          for i in range(min(self.num_vis, image_tensor.shape[0])):
            writer.add_image(k + f'/sample_{i}', image_tensor[i], epoch)
      
      if epoch % self.per_save_epochs == 0:
        self.save_checkpoint(epoch, is_tmp=False)
      
      if epoch % self.per_save_tmp_epochs == 0:
        self.save_checkpoint(epoch, is_tmp=True)
    
  def save_checkpoint(self, epoch, is_tmp=False):
    checkpoint = {
      'epoch': epoch,
      'model_state_dict': self.hamiltonian_net.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
    }
    ckpt_name = 'checkpoint.pth' if is_tmp else f'checkpoint_ep{epoch}.pth'
    ckpt_path = os.path.join(self.extract_dir, ckpt_name)
    torch.save(checkpoint, ckpt_path)

  def load_checkpoint(self, ckpt_name='checkpoint.pth'):
    ckpt_path = os.path.join(self.extract_dir, ckpt_name)
    checkpoint = torch.load(ckpt_path)
    self.hamiltonian_net.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch