"""Given the network and latent code, generate the trajectory.
"""

import os
import glob
import numpy as np
import torch

from input_pipeline import create_dataloader
from models_hamiltonian import get_model_hamiltoinian


class Generator(object):

  def __init__(self, config, generate_dir='gen_sequence', dataset_split='train', dtype=torch.float32):
    self.config = config
    self.dtype = dtype
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # general
    self.workdir = self.config.workdir
    self.generate_dir = os.path.join(self.workdir, generate_dir)

    # dataloaders
    self.train_loader = create_dataloader(config.data, split=dataset_split, shuffle=False)
    #self.test_loader = create_dataloader(config.data, split='test', shuffle=False)

    # networks
    self.hamiltonian_net = get_model_hamiltoinian(config.model, dtype=dtype)
    self.hamiltonian_net = self.hamiltonian_net.to(self.dtype).to(self.device)
    self.load_pretrained_checkpoint()
  
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
    self.hamiltonian_net.load_state_dict(checkpoint['model_state_dict'])
  
  def generate(self):
    self.hamiltonian_net.eval()
    for gen_config in self.config.gen_config_list:
      dict_results_all, dict_cond_all = [], []
      for i, data in enumerate(self.train_loader):
        data = self.preprocess_data(data)
        with torch.no_grad():
          dict_results = self.hamiltonian_net.gen_results_for_eval(data, gen_config)
        dict_results_all.append(dict_results)

        cond_dict = data['cond_dict']
        for k in cond_dict:
          cond_dict[k] = cond_dict[k].detach().cpu().numpy()
        dict_cond_all.append(data['cond_dict'])
      
      dict_results_dataset, dict_cond_dataset = {}, {}
      for k in dict_results_all[0]:
        dict_results_dataset[k] = np.concatenate(
          [dict_results[k] for dict_results in dict_results_all], axis=0)
      for k in dict_cond_all[0]:
        dict_cond_dataset[k] = np.concatenate(
          [dict_cond[k] for dict_cond in dict_cond_all], axis=0)
      
      save_dir = os.path.join(self.generate_dir, gen_config.name)
      os.makedirs(save_dir, exist_ok=True)
      np.save(os.path.join(save_dir, 'result_dict.npy'), dict_results_dataset)
      np.save(os.path.join(save_dir, 'cond_dict.npy'), dict_cond_dataset)