def get_model(model_config):
  
  if model_config.name == 'simple_transformer':
    from .simple_transformer import SimpleTransformer
    model = SimpleTransformer(model_config)
  
  elif model_config.name == 'baseline_hnn_resnet':
    from .baseline_hnn_resnet import SimpleResNet
    model = SimpleResNet(model_config)
  
  elif model_config.name == 'baseline_hnn_transformer':
    from .baseline_hnn_transformer import SimpleTransformer
    model = SimpleTransformer(model_config)
  
  elif model_config.name == 'baseline_vanilla_resnet':
    from .baseline_vanilla_resnet import SimpleResNet
    model = SimpleResNet(model_config)
  
  elif model_config.name == 'baseline_vanilla_transformer':
    from .baseline_vanilla_transformer import SimpleTransformer
    model = SimpleTransformer(model_config)
  
  elif model_config.name == 'baseline_cnn':
    from .baseline_cnn import SimpleCNN
    model = SimpleCNN(model_config)
  else:
    raise NotImplementedError('Model not implemented.')
  
  return model