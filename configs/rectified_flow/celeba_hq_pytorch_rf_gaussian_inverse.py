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
  training.snapshot_freq = 100000
  training.data_dir = '/home/yasmin/projects/stargan-v2/data/celeba_hq/val/subset5' # subset5: 5 images, subset100: 100 images

  # sampling
  sampling = config.sampling
  sampling.method = 'rectified_flow'
  sampling.init_type = 'gaussian' 
  sampling.init_noise_scale = 1.0
  sampling.use_ode_sampler = 'euler'
  sampling.sample_N = 100

  # inverse problem module
  eval = config.eval
  eval.enable_inverse = True
  eval.task = 'super_resolution'
  eval.operator = {'name': 'super_resolution', 'in_shape': (1, 3, 256, 256), 'scale_factor': 4}
  eval.noise = {'name': 'gaussian', 'sigma': 0.01}
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
  eval.nita = 10.
  eval.stop_time = 0.9
  eval.alpha = 1.
  eval.beta = 1.
  eval.eta_scheduler = 'constant'
  

  # data
  data = config.data
  data.dataset = 'CelebA-HQ-Pytorch'
  data.centered = True

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

  return config
