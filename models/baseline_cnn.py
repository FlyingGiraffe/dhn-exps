import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
  
  def __init__(self, model_config):
    super(SimpleCNN, self).__init__()

    # Hyperparameters
    self.q_dim = model_config.q_dim
    input_dim = self.q_dim * 2
    hidden_dim = model_config.hidden_dim
    self.num_stages = model_config.num_stages
    
    # Network layers
    self.conv_in = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
    self.conv_layers = nn.ModuleList()
    for i in range(self.num_stages):
      conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
      self.conv_layers.append(conv)
    self.conv_out = nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)

    self.act = nn.ReLU()

  def forward(self, q, p):
    x = torch.cat([q, p], dim=-1)
    x = x.permute(0, 2, 1)

    x = self.conv_in(x)
    for i in range(self.num_stages):
      num_steps = x.shape[2]
      num_steps_tgt = (num_steps - 1) * 2 + 1
      x = F.interpolate(x, size=num_steps_tgt, mode='linear', align_corners=True)
      x = self.act(self.conv_layers[i](x))
    x = self.conv_out(x)

    x = x.permute(0, 2, 1)
    q = x[:, :, :self.q_dim]
    p = x[:, :, self.q_dim:]

    return q, p