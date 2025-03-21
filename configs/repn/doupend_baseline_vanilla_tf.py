"""Representation Learning, Double Pendulum.
Baseline: Vanilla NN.
"""

import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  config.workdir = 'tmp'

  config.data = data = ml_collections.ConfigDict()
  data.path = 'data/double_pendulum'
  data.batch_size = 32
  data.num_workers = 2
  data.prefetch_factor = 2
  data.pin_memory = False
  data.cache = True

  config.model = model = ml_collections.ConfigDict()
  # hamiltonian
  model.hamiltonian = 'baseline_vanilla_repn'
  # general features
  model.num_embeddings = 1000
  model.embedding_dim = 128
  model.q_dim = 2
  model.c_dim = 1
  model.t_span = (0, 10)
  model.step_size = 8
  # codebook
  model.codebook = codebook = ml_collections.ConfigDict()
  codebook.num_embeddings = model.num_embeddings
  codebook.embedding_dim = model.embedding_dim
  codebook.normalize_emb = True
  # hamiltonian
  model.network = network = ml_collections.ConfigDict()
  network.name = 'baseline_vanilla_transformer'
  network.q_dim = model.q_dim
  network.z_dim = model.embedding_dim
  network.output_dim = model.q_dim
  network.hidden_dim = 128
  network.num_heads = 4
  network.num_layers = 2
  
  config.loss = loss = ml_collections.ConfigDict()
  loss.weight_eom = 1.0
  loss.crop_interval = (0, -1)

  config.optim = optim = ml_collections.ConfigDict()
  optim.num_epochs = 200
  optim.lr = 1e-4

  config.logging = logging = ml_collections.ConfigDict()
  logging.per_save_epochs = 50
  logging.per_save_tmp_epochs = 1
  logging.per_eval_epochs = 1
  logging.num_eval_batches = 1
  logging.num_vis = 8

  return config