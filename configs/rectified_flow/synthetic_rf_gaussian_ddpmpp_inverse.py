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
"""Training Rectified Flow on CIFAR-10 with DDPM++."""

from configs.default_synthetic_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'rectified_flow'
  training.continuous = False
  training.snapshot_freq = 200 # could be 100k 
  training.reduce_mean = True

  # sampling
  sampling = config.sampling
  sampling.method = 'rectified_flow'
  sampling.init_type = 'gaussian' 
  sampling.init_noise_scale = 1.0
  sampling.use_ode_sampler = 'euler' ### rk45 or euler
  sampling.ode_tol = 1e-5

  # data
  data = config.data
  data.centered = False


   # inverse problem module
  eval = config.eval
  eval.begin_ckpt = 8
  eval.end_ckpt = 8
  eval.gap_ckpt = 1
  eval.enable_inverse = True
  eval.enable_sampling = False
  eval.task = 'noise'
  eval.operator = {'name': 'noise', }
  eval.noise = {'name': 'gaussian', 'sigma': 0.1}
  eval.mask_opt = {}
  eval.eta = 1.0 # step size of the gradient descent
  eval.batch_size = 1
  eval.init = 0.0
  eval.method = 'ours_xx' # ['naive', 'pseudo', 'mc']
  eval.compute_fid = False
  eval.lamda = 0.
  eval.k = 1
  eval.n_trace = 1
  eval.zeta = 1.
  eval.nita = 1.
  eval.seed = 1126
  eval.number = 5
  

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.999999
  model.dropout = 0.15
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'none'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.embedding_type = 'positional'
  model.fourier_scale = 16
  model.conv_size = 3

  return config
