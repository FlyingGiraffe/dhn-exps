import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  config.simulator = 'single_pendulum'

  config.save_dir = 'tmp'
  config.num_data = 1000
  config.num_vis = 5

  config.t_span = (0, 10)
  config.num_timesteps = 1025

  config.constants = constants = ml_collections.ConfigDict()
  constants.g = 9.81  # Gravity [m/s^2]
  constants.m = 1.0  # Mass of the pendulum [kg]
  constants.l = (0.5, 1.0)  # Length of the pendulum

  return config