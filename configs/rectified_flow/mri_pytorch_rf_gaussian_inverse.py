# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training rectified Flow on CelebA HQ."""

from configs.default_lsun_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training 
  training = config.training
  training.sde = 'rectified_flow'
  training.continuous = False
  training.reduce_mean = True
  training.snapshot_freq = 2000
  training.data_dir = 'train'
  training.eval_dir = 'test'
  training.n_iters = 1300001
  training.log_freq = 100
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.batch_size = 50

  eval = config.eval
  eval.enable_inverse = True
  eval.task = 'mri'
  eval.operator = {'name': 'mri', 'in_shape': (1, 3, 128, 128), 'scale_factor': 4}
  eval.noise = {'name': 'gaussian', 'sigma': 0.00}
  eval.mask_opt = {}
  eval.eta = 1.0 # step size of the gradient descent
  eval.batch_size = 1
  eval.init = 0.0
  eval.method = 'naive' # ['naive', 'pseudo', 'mc']
  eval.compute_fid = False
  eval.lamda = 100.0
  eval.k = 1
  eval.n_trace = 1
  eval.zeta = 1.
  eval.nita = 1
  eval.stop_time = 1
  eval.alpha = 1.
  eval.beta = 1.
  eval.eta_scheduler = 'constant'
  eval.rate = 4
 

  # sampling
  sampling = config.sampling
  sampling.method = 'rectified_flow'
  sampling.init_type = 'gaussian' 
  sampling.init_noise_scale = 1.0
  sampling.use_ode_sampler = 'euler'
  sampling.sample_N = 100

  # data
  data = config.data
  data.dataset = 'mri-Pytorch'

  data.image_size = 128
  data.random_flip = False
  data.centered = True
  data.uniform_dequantization = False
  data.num_channels = 1

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  optim = config.optim 
 
  optim.lr = 1e-4 # original: 2e-4
  optim.warmup = 2000

  return config
