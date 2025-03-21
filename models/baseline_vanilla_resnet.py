import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
  def __init__(self, input_dim, output_dim, num_groups=16):
    super(ResidualBlock, self).__init__()
    self.output_dim = output_dim

    self.linear_residual = nn.Linear(input_dim, output_dim)

    self.linear1 = nn.Linear(input_dim, output_dim)
    self.linear2 = nn.Linear(output_dim, output_dim)

    self.norm1 = nn.GroupNorm(num_channels=output_dim, num_groups=num_groups)
    self.norm2 = nn.GroupNorm(num_channels=output_dim, num_groups=num_groups)

    self.act = nn.ReLU()

  def forward(self, x):
    residual = self.linear_residual(x)

    x = self.linear1(x)
    #x = self.norm1(x[None]).squeeze(0)
    x = self.act(x)

    x = self.linear2(x)
    #x = self.norm2(x[None]).squeeze(0)
    x = self.act(x)

    return x + residual


class SimpleResNet(nn.Module):
  
  def __init__(self, model_config):
    super(SimpleResNet, self).__init__()

    # Hyperparameters

    self.q_dim = model_config.q_dim
    z_dim = model_config.z_dim
    input_dim = self.q_dim * 2 + z_dim
    output_dim = model_config.output_dim

    hidden_dim = model_config.hidden_dim
    self.num_res_blocks = model_config.num_res_blocks
    
    # Network layers

    self.fc_in = nn.Linear(input_dim, hidden_dim)

    self.res_blocks = nn.ModuleList()
    for i in range(self.num_res_blocks):
      res_block_i = ResidualBlock(
        input_dim=hidden_dim,
        output_dim=hidden_dim,
      )
      self.res_blocks.append(res_block_i)

    self.fc_out = nn.Linear(hidden_dim, output_dim)

    self.act = nn.ReLU()

  def forward(self, q, p, z):
    x = torch.cat([q, p, z], dim=-1)
    
    x = self.fc_in(x)
    for i in range(self.num_res_blocks):
      x = self.act(self.res_blocks[i](x))
    x = self.fc_out(x)

    q = x[:, :self.q_dim]
    p = x[:, self.q_dim:]

    return q, p