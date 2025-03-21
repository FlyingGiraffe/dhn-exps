import os
import pickle

import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader


class SimDataset(Dataset):

  def __init__(self, data_config, split=None, filename='data.pkl'):
    data_path = os.path.join(data_config.path, split, filename)
    
    with open(data_path, 'rb') as f:
      self.data = pickle.load(f)
    self.num_data = len(self.data)
  
  def __len__(self):
    return self.num_data

  def __getitem__(self, idx):
    data = self.data[idx]
    return data


def worker_init_fn(worker_id):
  seed = worker_id + 1000
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def create_dataloader(data_config, split, filename='data.pkl', shuffle=None):

  dataset = SimDataset(data_config, split, filename)

  dataloader = DataLoader(
    dataset, batch_size=data_config.batch_size,
    shuffle=(split == 'train') if shuffle is None else shuffle,
    drop_last=False,
    worker_init_fn=worker_init_fn,
    num_workers=data_config.num_workers,
    prefetch_factor=data_config.prefetch_factor if data_config.num_workers > 0 else None,
    pin_memory=data_config.pin_memory,
    persistent_workers=True if data_config.num_workers > 0 else False,
  )
  
  return dataloader