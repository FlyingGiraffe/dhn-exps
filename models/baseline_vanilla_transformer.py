import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class SimpleTransformer(nn.Module):
  
  def __init__(self, model_config):
    super(SimpleTransformer, self).__init__()

    # Hyperparameters

    q_dim = model_config.q_dim  # for the whole block
    z_dim = model_config.z_dim
    output_dim = model_config.output_dim

    hidden_dim = model_config.hidden_dim
    num_heads = model_config.num_heads
    self.num_layers = model_config.num_layers
    
    # Network layers

    self.q_embedding = nn.Linear(q_dim, hidden_dim)
    self.p_embedding = nn.Linear(q_dim, hidden_dim)
    self.z_embedding = nn.Linear(z_dim, hidden_dim)

    num_tokens = 3
    self.positional_embedding = nn.Parameter(torch.zeros(num_tokens, hidden_dim))

    self.layers = nn.ModuleList()
    for i in range(self.num_layers):
      layer_i = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=hidden_dim * 4,
        dropout=0.0,
        batch_first=True,  # only for this baseline network, the HNN networks are vmapped
      )
      self.layers.append(layer_i)

    self.q_fc_out = nn.Linear(hidden_dim, output_dim)
    self.p_fc_out = nn.Linear(hidden_dim, output_dim)

    self.act = nn.ReLU()

  def forward(self, q, p, z):

    q_emb = self.q_embedding(q)
    p_emb = self.p_embedding(p)
    z_emb = self.z_embedding(z)

    # get a token sequence of shape [1 + q_tokens + p_tokens, embedding_dim]
    x = torch.cat([z_emb[:, None], q_emb[:, None], p_emb[:, None]], dim=1)
    x = x + self.positional_embedding
    
    # disable the fused kernels for computing second-order derivative
    # if not disabled, will get the error:
    # RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented
    with sdpa_kernel(SDPBackend.MATH):
      for i in range(self.num_layers):
        x = self.layers[i](x)
    
    q_pred = x[:, 1]
    p_pred = x[:, 2]

    q_pred = self.q_fc_out(q_pred)
    p_pred = self.p_fc_out(p_pred)

    return q_pred, p_pred