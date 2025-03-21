import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class SimpleTransformer(nn.Module):
  
  def __init__(self, model_config):
    super(SimpleTransformer, self).__init__()

    # Hyperparameters

    q_dim = model_config.q_dim  # for the whole block
    z_dim = model_config.z_dim
    block_size = model_config.block_size
    output_dim = model_config.output_dim

    hidden_dim = model_config.hidden_dim
    num_heads = model_config.num_heads
    
    self.num_layers = model_config.num_layers
    self.num_noise_scales = model_config.num_noise_scales
    
    # Network layers

    self.q_embedding = nn.Linear(q_dim, hidden_dim)
    self.p_embedding = nn.Linear(q_dim, hidden_dim)
    self.z_embedding = nn.Linear(z_dim, hidden_dim)

    num_tokens = block_size * 2 + 1
    self.positional_embedding = nn.Parameter(torch.zeros(num_tokens, hidden_dim))
    self.noise_scale_embedding = nn.Embedding(self.num_noise_scales + 1, hidden_dim)

    self.layers = nn.ModuleList()
    for i in range(self.num_layers):
      layer_i = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=hidden_dim * 4,
        dropout=0.0,
      )
      self.layers.append(layer_i)

    self.fc_out = nn.Linear(hidden_dim, output_dim)

    self.act = nn.ReLU()

  def forward(self, q, p, q_mask, p_mask, z):

    q_emb = self.q_embedding(q)
    p_emb = self.p_embedding(p)
    z_emb = self.z_embedding(z)

    # assume all masks are for the temporal dimension not the spatial dimension
    q_mask, p_mask = q_mask[..., 0], p_mask[..., 0]

    # get a token sequence of shape [1 + q_tokens + p_tokens, embedding_dim]
    x = torch.cat([z_emb[None], q_emb, p_emb], dim=0)
    x = x + self.positional_embedding

    # assuming a soft mask, with t in [0, 1] indicating noise scales
    z_mask = torch.ones(1,).to(z.device)
    mask = torch.cat([z_mask, q_mask, p_mask], dim=0)
    mask_emb = (mask * self.num_noise_scales).to(torch.int64)
    a_emb = self.noise_scale_embedding(mask_emb)
    x = x + a_emb
    
    # disable the fused kernels for computing second-order derivative
    # if not disabled, will get the error:
    # RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented
    with sdpa_kernel(SDPBackend.MATH):
      for i in range(self.num_layers):
        x = self.layers[i](x)
    
    x = self.fc_out(x)
    x = x[0]

    return x