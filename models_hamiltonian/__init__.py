def get_model_hamiltoinian(model_config, dtype):
  
  if model_config.hamiltonian == 'default':
    from .hamiltonian import HamiltonianNet
    model = HamiltonianNet(model_config, dtype=dtype)
  
  ##################################################
  # AutoRegression
  ##################################################

  elif model_config.hamiltonian == 'ar':
    from .hamiltonian_ar import HamiltonianNet
    model = HamiltonianNet(model_config, dtype=dtype)
  
  elif model_config.hamiltonian == 'baseline_hnn_ar':
    from .baseline_hnn_ar import HamiltonianNet
    model = HamiltonianNet(model_config, dtype=dtype)
  
  elif model_config.hamiltonian == 'baseline_vanilla_ar':
    from .baseline_vanilla_ar import HamiltonianNet
    model = HamiltonianNet(model_config, dtype=dtype)
  
  ##################################################
  # Representation Learning
  ##################################################
  
  elif model_config.hamiltonian == 'repn':
    from .hamiltonian_repn import HamiltonianNet
    model = HamiltonianNet(model_config, dtype=dtype)
  
  elif model_config.hamiltonian == 'baseline_hnn_repn':
    from .baseline_hnn_repn import HamiltonianNet
    model = HamiltonianNet(model_config, dtype=dtype)

  elif model_config.hamiltonian == 'baseline_vanilla_repn':
    from .baseline_vanilla_repn import HamiltonianNet
    model = HamiltonianNet(model_config, dtype=dtype)
  
  ##################################################
  # Super-Resolution
  ##################################################
  
  elif model_config.hamiltonian == 'superres':
    from .hamiltonian_superres import HamiltonianNet
    model = HamiltonianNet(model_config, dtype=dtype)

  elif model_config.hamiltonian == 'baseline_cnn_superres':
    from .baseline_cnn_superres import HamiltonianNet
    model = HamiltonianNet(model_config, dtype=dtype)

  else:
    raise NotImplementedError('Hamiltonian model not implemented.')
  
  return model