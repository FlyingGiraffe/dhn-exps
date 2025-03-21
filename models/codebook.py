import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(nn.Module):
  def __init__(self, model_cfg):
    """
    Initializes the codebook with randomly initialized embeddings.
    
    Args:
      num_embeddings (int): Number of discrete codebook embeddings.
      embedding_dim (int): Dimensionality of each embedding vector.
    """
    super(Codebook, self).__init__()
    embedding_dim = model_cfg.embedding_dim
    num_embeddings = model_cfg.num_embeddings
    self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

    self.normalize_emb = model_cfg.normalize_emb
      
  def forward(self, idx):
    emb = self.embeddings(idx)
    if self.normalize_emb:
      emb = F.normalize(emb, dim=-1)
    return emb