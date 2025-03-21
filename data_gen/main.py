from absl import app
from absl import flags
import os

import numpy as np
from ml_collections.config_flags import config_flags
import pickle

from simulators import get_simulator


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.mark_flags_as_required(['config'])


def main(argv):
  config = FLAGS.config

  np.random.seed(FLAGS.seed)

  trajectories = []
  for i_data in range(config.num_data):
    simulator = get_simulator(config)
    traj = simulator.sample_trajectory()
    traj['idx'] = i_data
    trajectories.append(traj)

    if i_data < config.num_vis:
      simulator.visualize(traj, i_data)

  os.makedirs(config.save_dir, exist_ok=True)
  with open(os.path.join(config.save_dir, 'data.pkl'), 'wb') as fp:
    pickle.dump(trajectories, fp)


if __name__ == '__main__':
  app.run(main)  