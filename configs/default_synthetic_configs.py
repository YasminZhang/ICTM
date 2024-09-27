import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 128
  training.n_iters = 10000
  training.snapshot_freq = 200
  training.log_freq = 200
  training.eval_freq = 200
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 200
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16
  
  sampling.sigma_variance = 0.0 # NOTE: sigma variance for turning ODE to SDE
  sampling.init_noise_scale = 1.0
  sampling.use_ode_sampler = 'euler'
  sampling.ode_tol = 1e-5
  sampling.sample_N = 100

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 22
  evaluate.end_ckpt = 22
  evaluate.gap_ckpt = 1
  evaluate.batch_size = 128
  evaluate.enable_sampling = True
  evaluate.num_samples = 128
  evaluate.enable_loss = False
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'Gaussian'
  data.image_size = 16
  data.random_flip = False
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 1

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0.
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config
