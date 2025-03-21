"""Representation Learning, Single Pendulum.
Baseline: Vanilla NN.
"""

import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  config.workdir = 'tmp'

  config.data = data = ml_collections.ConfigDict()
  data.path = 'data/single_pendulum'
  data.batch_size = 32
  data.num_workers = 2
  data.prefetch_factor = 2
  data.pin_memory = False
  data.cache = True

  config.model = model = ml_collections.ConfigDict()
  # hamiltonian
  model.hamiltonian = 'baseline_cnn_superres'
  # general features
  model.num_embeddings = 1000
  model.q_dim = 1
  model.t_span = (0, 10)
  model.stage_step_size = (32, 8)
  model.step_size = 8
  model.train_step_span = (0, 513)
  # hamiltonian
  model.network = network = ml_collections.ConfigDict()
  network.name = 'baseline_cnn'
  network.q_dim = model.q_dim
  network.hidden_dim = 128
  network.num_stages = 2
  
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