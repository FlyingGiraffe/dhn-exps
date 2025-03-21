def get_simulator(config):

  if config.simulator == 'single_pendulum':
    from .single_pendulum import Simulator

  elif config.simulator == 'double_pendulum':
    from .double_pendulum import Simulator
  
  else:
    raise NotImplementedError('Simulator not implemented.')
  
  return Simulator(config)