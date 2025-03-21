"""AutoRegression, Single Pendulum.
Baseline: Vanilla NN.
"""

import ml_collections


def get_gen_config():
  gen_config = ml_collections.ConfigDict()
  gen_config.name = 'vanilla'
  gen_config.num_init_steps = 8
  return gen_config


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
  model.hamiltonian = 'baseline_vanilla_ar'
  # general features
  model.num_embeddings = 1000
  model.embedding_dim = 128
  model.q_dim = 1
  model.t_span = (0, 10)
  model.step_size = 8
  # codebook
  model.codebook = codebook = ml_collections.ConfigDict()
  codebook.num_embeddings = model.num_embeddings
  codebook.embedding_dim = model.embedding_dim
  codebook.normalize_emb = False
  # hamiltonian
  model.network = network = ml_collections.ConfigDict()
  network.name = 'baseline_vanilla_resnet'
  network.q_dim = model.q_dim
  network.z_dim = model.embedding_dim
  network.output_dim = model.q_dim * 2
  network.hidden_dim = 128
  network.num_res_blocks = 1
  
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

  config.gen_config_list = (
    get_gen_config(),
  )

  return config