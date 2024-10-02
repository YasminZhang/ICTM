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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """
from functools import partial
import gc
import io
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint

from inverse.measurements import get_operator, get_noise, get_cs_A
from inverse.img_utils import clear_color, mask_generator, psnr_fn, normalize_torch, unnormalize_torch
import matplotlib.pyplot as plt

from cleanfid import fid as fid_fn
import lpips
from pytorch_msssim import ssim

# import torchvision
import torchvision





FLAGS = flags.FLAGS


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  train_ds, eval_ds = datasets.get_pytorch_dataset(config)

  train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, drop_last=True, num_workers=4)
  eval_ds_loader = torch.utils.data.DataLoader(eval_ds, batch_size=config.training.batch_size, shuffle=True, drop_last=True, num_workers=4)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'rectified_flow':
    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  step = initial_step - 1
  while step <= (num_train_steps + 1):
    for _, data in enumerate(train_ds_loader):
        step += 1
        batch = data.to(config.device).float()
        batch = scaler(batch)
    
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
          logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
          writer.add_scalar("training_loss", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
          save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
          for _, eval_data in enumerate(eval_ds_loader):  
            break  
          eval_batch = eval_data.to(config.device).float()
          eval_batch = scaler(eval_batch)
          eval_loss = eval_step_fn(state, eval_batch)
          logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
          writer.add_scalar("eval_loss", eval_loss.item(), step)

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
          # Save the checkpoint.
          save_step = step // config.training.snapshot_freq
          save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

          # Generate and save samples
          if config.training.snapshot_sampling:
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            sample, n = sampling_fn(score_model)
            ema.restore(score_model.parameters())
            this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
            tf.io.gfile.makedirs(this_sample_dir)
            nrow = int(np.sqrt(sample.shape[0]))
            image_grid = make_grid(sample, nrow, padding=2)
           
            sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8) # TODO: check this.
            with tf.io.gfile.GFile(
                os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
              np.save(fout, sample)

            with tf.io.gfile.GFile(
                os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
              save_image(image_grid, fout)


def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Build data pipeline
  #train_ds, eval_ds, _ = datasets.get_dataset(config,
  #                                            uniform_dequantization=config.data.uniform_dequantization,
  #                                            evaluation=True)
  train_ds, eval_ds = datasets.get_pytorch_dataset(config)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'rectified_flow':
    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler, sigma_var=config.sampling.sigma_variance, ode_tol=config.sampling.ode_tol, sample_N=config.sampling.sample_N)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting)


  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  #train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
  #                                                    uniform_dequantization=True, evaluation=True)
  train_ds_bpd, eval_ds_bpd = datasets.get_pytorch_dataset(config)   ###NOTE: XC: fix later

  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = 5
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enabled
  if (config.eval.enable_sampling) or (config.eval.enable_figures_only):
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    while not tf.io.gfile.exists(ckpt_filename):
      if not waiting_message_printed:
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        waiting_message_printed = True
      time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(60)
      try:
        state = restore_checkpoint(ckpt_path, state, device=config.device)
      except:
        time.sleep(120)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      all_losses = []
      eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for i, batch in enumerate(eval_iter):
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if (i + 1) % 1000 == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)
      with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in range(len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
          eval_batch = eval_batch.permute(0, 3, 1, 2)
          eval_batch = scaler(eval_batch)
          bpd = likelihood_fn(score_model, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                              f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                 "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())
    
    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
      for r in range(num_sampling_rounds):
        logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = os.path.join(
          eval_dir, f"ckpt_{ckpt}")
        tf.io.gfile.makedirs(this_sample_dir)
        samples, n = sampling_fn(score_model)
        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())

        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(
            io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
          fout.write(io_buffer.getvalue())

      # Compute inception scores, FIDs and KIDs.
      # Load all statistics that have been previously computed and saved for each host
      all_logits = []
      all_pools = []
      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
      stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
      for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
          stat = np.load(fin)
          if not inceptionv3:
            all_logits.append(stat["logits"])
          all_pools.append(stat["pool_3"])

      if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
      all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

      # Load pre-computed dataset statistics.
      data_stats = evaluation.load_dataset_stats(config)
      data_pools = data_stats["pool_3"]

      # Compute FID/KID/IS on all samples together.
      if not inceptionv3:
        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
      else:
        inception_score = -1

      fid = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, all_pools)
      # Hack to get tfgan KID work for eager execution.
      tf_data_pools = tf.convert_to_tensor(data_pools)
      tf_all_pools = tf.convert_to_tensor(all_pools)
      kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools).numpy()
      del tf_data_pools, tf_all_pools

      logging.info(
        "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
          ckpt, inception_score, fid, kid))

      with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                             "wb") as f:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
        f.write(io_buffer.getvalue())
   

    if config.eval.enable_figures_only:
      import torchvision
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
      for r in range(num_sampling_rounds):
        logging.info("sampling only figures -- ckpt: %d, round: %d" % (ckpt, r))
        this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
        
        # Directory to save samples. Different for each host to avoid writing conflicts
        samples, n = sampling_fn(score_model)
        torchvision.utils.save_image(samples.clamp_(0.0, 1.0), os.path.join(this_sample_dir, '%d.png'%r), nrow=10, normalize=False)


def evaluate_inverse(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """

  if config.eval.task == 'super_resolution' or config.eval.task == 'super_resolution1':
    config.eval.operator = {'name': 'super_resolution', 'in_shape': (1, 3, 256, 256), 'scale_factor': 4}
  elif config.eval.task == 'inpainting':
    config.eval.operator = {'name': 'inpainting'}
    config.eval.mask_opt = {'mask_type': 'random', 'mask_prob_range': (0.7, 0.7), 'image_size': 256}
  elif config.eval.task == 'inpainting_box':
    config.eval.operator = {'name': 'inpainting'}
    config.eval.mask_opt = {'mask_type': 'box', 'mask_len_range': (128, 129) , 'image_size': 256}
  elif config.eval.task == 'gaussian':
    config.eval.operator = {'name': 'gaussian_blur', 'kernel_size': 61, 'intensity': 3.0}
  elif config.eval.task == 'nonlinear':
    config.eval.operator = {'name': 'nonlinear_blur', 'opt_yml_path': '/home/yasmin/projects/RectifiedFlow/ImageGeneration/inverse/bkse/options/generate_blur/default.yml'}
  elif config.eval.task == 'motion':
    config.eval.operator = {'name': 'motion_blur', 'kernel_size': 61, 'intensity': 0.5}
  elif config.eval.task == 'phase':
    config.eval.operator = {'name': 'phase_retrieval', 'oversample': 2.0}
  elif config.eval.task == 'mri':
    config.eval.operator = {'name': 'mri', 'in_shape': 128,}
  elif config.eval.task == 'cs':
    config.eval.operator = {'name': 'cs', }
    config.eval.noise = {'name': 'gaussian', 'sigma': 0.001}
    # n_measurements, n_input, A, sign_pattern, indices
    
    config.eval.operator['n_input'] = 128*128
    config.eval.operator['n_measurements'] = int(128*128/config.eval.rate) 
    config.eval.operator['A'] = np.random.randn( config.eval.operator['n_input'])
    config.eval.operator['sign_pattern'] = np.random.randint(0, 2, config.eval.operator['n_input']) * 2 - 1
    config.eval.operator['indices'] = np.random.choice(config.eval.operator['n_input'], config.eval.operator['n_measurements'], replace=False)
  else:
    raise ValueError(f"Task {config.eval.task} not recognized.")


 
  # Sampling configuration
  logging.info(f"ODE sampler used: {config.sampling.use_ode_sampler}")
  logging.info(f"Number of samples: {config.sampling.sample_N}")
  logging.info(f"eta: {config.eval.eta}")

  # Inverse problem evaluation configuration
  logging.info(f"Batch size for evaluation: {config.eval.batch_size}")
  logging.info(f"Initial value for evaluation: {config.eval.init}")
  logging.info(f"Method for evaluation: {config.eval.method}")
  



  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Build data pipeline
  #train_ds, eval_ds, _ = datasets.get_dataset(config,
  #                                            uniform_dequantization=config.data.uniform_dequantization,
  #                                            evaluation=True)
  train_ds, eval_ds = datasets.get_pytorch_dataset(config)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'rectified_flow':
    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler, sigma_var=config.sampling.sigma_variance, ode_tol=config.sampling.ode_tol, sample_N=config.sampling.sample_N)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting)


  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  #train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
  #                                                    uniform_dequantization=True, evaluation=True)
  if config.eval.enable_bpd:
    train_ds_bpd, eval_ds_bpd = datasets.get_pytorch_dataset(config)   ###NOTE: XC: fix later

    if config.eval.bpd_dataset.lower() == 'train':
      ds_bpd = train_ds_bpd
      bpd_num_repeats = 1
    elif config.eval.bpd_dataset.lower() == 'test':
      # Go over the dataset 5 times when computing likelihood on the test dataset
      ds_bpd = eval_ds_bpd
      bpd_num_repeats = 5
    else:
      raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enabled
  if (config.eval.enable_sampling) or (config.eval.enable_figures_only):
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
  
  if config.eval.enable_inverse:
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, inverse=True)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    while not tf.io.gfile.exists(ckpt_filename):
      if not waiting_message_printed:
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        waiting_message_printed = True
      time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(60)
      try:
        state = restore_checkpoint(ckpt_path, state, device=config.device)
      except:
        time.sleep(120)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())


    
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      all_losses = []
      eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for i, batch in enumerate(eval_iter):
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if (i + 1) % 1000 == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)
      with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())


    if config.eval.enable_inverse:
      # inverse operator

      for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(eval_dir, img_dir), exist_ok=True)
       
      operator = get_operator(device=config.device, **config.eval.operator)
      noiser = get_noise(**config.eval.noise)

      if config.eval.operator['name'] == 'inpainting':
        mask_gen = mask_generator(
        **config.eval.mask_opt
        )

      psnrs = []
      lpipss = []
      ssims = []
      loss_fn_alex = lpips.LPIPS(net='vgg').to(config.device) # best forward scores
      
      bpd_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for batch_id in range(len(eval_ds)):
        # if batch_id == 1:
        #   break
        j = batch_id
        eval_batch = next(bpd_iter)
        eval_batch = eval_batch[None, ...].to(config.device).float() # shape: [1, 3, 256, 256]
        eval_batch = scaler(eval_batch)  # [0, 1] -> [-1, 1]
        fname = str(j).zfill(5) + '.png'
 
        mask = None
        # Exception) In case of inpainting 
        if config.eval.operator ['name'] == 'inpainting':
          mask = mask_gen(eval_batch)
          mask = mask[:, 0, :, :].unsqueeze(dim=0)
          
          # Forward measurement model (Ax + n)
          y = operator.forward(eval_batch, mask=mask)
          y_n = noiser(y)

        elif config.eval.operator['name'] == 'mri':
          import torchvision
          y, imgs_to_save = operator.forward(eval_batch, return_img=True)
          for j, tmp_img in enumerate(imgs_to_save):
              tmp_img[2] = noiser(tmp_img[2])
              torchvision.utils.save_image(tmp_img[[2,3]], os.path.join(eval_dir, 'input', fname), normalize=False, nrow=2)
          y_n = noiser(y)
        elif config.eval.operator['name'] == 'cs':
         
          y = operator.forward(eval_batch  )
         
          y_n = noiser(y)
         
        else: 
          # Forward measurement model (Ax + n)
          y = operator.forward(eval_batch)
          y_n = noiser(y)
        
        if config.eval.method == 'ls':
          # Least square solution
          sample = operator.ls(y_n)
          print(y_n.shape, sample.shape)
          sample = sample - sample.min()
          sample = sample / sample.max()

          eval_batch = eval_batch - eval_batch.min()
          eval_batch = eval_batch / eval_batch.max()
          # plt.imsave(os.path.join(eval_dir, 'input', fname), y_n[0, 0].cpu().numpy(), cmap='gray')
          plt.imsave(os.path.join(eval_dir, 'label', fname), eval_batch[0, 0].cpu().numpy(), cmap='gray')
          plt.imsave(os.path.join(eval_dir, 'recon', fname), sample.cpu().numpy(), cmap='gray')
          return 
        else:
        
 
        # Sampling
          if config.eval.init == 0.0:
            x_start = torch.randn(eval_batch.shape, device=config.device) 
          else:
            if config.eval.operator['name'] == 'cs':
              x_start = config.eval.init * operator.transpose(y_n)  +  (1 - config.eval.init) * torch.randn(eval_batch.shape, device=config.device)
              print(x_start.shape, y_n.shape, eval_batch.shape)
            else:
              x_start = config.eval.init * operator.transpose(y_n)  +  (1 - config.eval.init) * torch.randn(eval_batch.shape, device=config.device) 
          sample, nfe = sampling_fn(model=score_model, z=x_start, measurement=y_n, task=config.eval.task,  init=config.eval.init,  operator=operator, record=False, save_root=eval_dir, 
                                    n_trace=config.eval.n_trace,alpha = config.eval.alpha, eta_scheduler=config.eval.eta_scheduler,  beta=config.eval.beta, zeta=config.eval.zeta, num_iter=config.eval.k, lamda = config.eval.lamda, stop_time=config.eval.stop_time,  eta=config.eval.eta, method=config.eval.method, nita=config.eval.nita,  mask=mask)

        # clear_color use x - x.min() / x.max() - x.min() to make the image in [0, 1]

        
        if config.eval.operator['name'] == 'mri' or config.eval.operator['name'] == 'cs':
          # y_n = y_n - y_n.min()
          # y_n = y_n / y_n.max()
        
          sample = sample - sample.min()
          sample = sample / sample.max()

          eval_batch = eval_batch - eval_batch.min()
          eval_batch = eval_batch / eval_batch.max()
          # plt.imsave(os.path.join(eval_dir, 'input', fname), y_n[0, 0].cpu().numpy(), cmap='gray')
          plt.imsave(os.path.join(eval_dir, 'label', fname), eval_batch[0, 0].cpu().numpy(), cmap='gray')
          plt.imsave(os.path.join(eval_dir, 'recon', fname), sample[0, 0].cpu().numpy(), cmap='gray')
        else:

          plt.imsave(os.path.join(eval_dir, 'input', fname), clear_color(y_n))
          plt.imsave(os.path.join(eval_dir, 'label', fname), clear_color(eval_batch))
          plt.imsave(os.path.join(eval_dir, 'recon', fname), clear_color(sample))


        c, d = unnormalize_torch(*normalize_torch(eval_batch, sample)) # clear color
        c, d = eval_batch, sample


        # PNSR
        psnr = psnr_fn(c, d)
        logging.info(f"batch: {batch_id}, PSNR: {psnr}")
        psnrs.append(psnr)
        # SSIM
        
        ssim_val = ssim((c+1)/2, (d+1)/2, data_range=1, size_average=True).item()
        logging.info(f"batch: {batch_id}, SSIM: {ssim_val}")
        ssims.append(ssim_val)
        # LPIPS
        lpips_val = loss_fn_alex(c, d).mean().item()
        lpipss.append(lpips_val)
        logging.info(f"batch: {batch_id}, LPIPS: {lpips_val}")
      
      # report PSNR, SSIM, LPIPS
      logging.info(f"ckpt: {ckpt}, LPIPS: {np.mean(lpipss):.3f} +- {np.std(lpipss):.2f}")
      logging.info(f"ckpt: {ckpt}, PSNR: {np.mean(psnrs):.2f} +- {np.std(psnrs):.2f}")
      logging.info(f"ckpt: {ckpt}, SSIM: {np.mean(ssims):.3f} +- {np.std(ssims):.2f}")
      # report LPIPS, PSNR, SSIM in one line
      logging.info(f"ckpt: {ckpt},  {np.mean(lpipss):.3f} +- {np.std(lpipss):.2f}  {np.mean(psnrs):.2f} +- {np.std(psnrs):.2f} {np.mean(ssims):.3f} +- {np.std(ssims):.2f}")
      


      if config.eval.compute_fid:
        fid_score = fid_fn.compute_fid(eval_dir + '/label', eval_dir + '/recon')
        # report FID
        logging.info(f"ckpt: {ckpt}, FID: {fid_score}")



    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in range(len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
          eval_batch = eval_batch.permute(0, 3, 1, 2)
          eval_batch = scaler(eval_batch)
          bpd = likelihood_fn(score_model, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                              f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                 "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())
    
    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
      for r in range(num_sampling_rounds):
        logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = os.path.join(
          eval_dir, f"ckpt_{ckpt}")
        tf.io.gfile.makedirs(this_sample_dir)
        samples, n = sampling_fn(score_model)
        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())

        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(
            io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
          fout.write(io_buffer.getvalue())

      # Compute inception scores, FIDs and KIDs.
      # Load all statistics that have been previously computed and saved for each host
      all_logits = []
      all_pools = []
      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
      stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
      for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
          stat = np.load(fin)
          if not inceptionv3:
            all_logits.append(stat["logits"])
          all_pools.append(stat["pool_3"])

      if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
      all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

      # Load pre-computed dataset statistics.
      data_stats = evaluation.load_dataset_stats(config)
      data_pools = data_stats["pool_3"]

      # Compute FID/KID/IS on all samples together.
      if not inceptionv3:
        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
      else:
        inception_score = -1

      fid = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, all_pools)
      # Hack to get tfgan KID work for eager execution.
      tf_data_pools = tf.convert_to_tensor(data_pools)
      tf_all_pools = tf.convert_to_tensor(all_pools)
      kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools).numpy()
      del tf_data_pools, tf_all_pools

      logging.info(
        "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
          ckpt, inception_score, fid, kid))

      with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                             "wb") as f:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
        f.write(io_buffer.getvalue())
   

    if config.eval.enable_figures_only:
      import torchvision
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
      for r in range(num_sampling_rounds):
        logging.info("sampling only figures -- ckpt: %d, round: %d" % (ckpt, r))
        this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
        
        # Directory to save samples. Different for each host to avoid writing conflicts
        samples, n = sampling_fn(score_model)
        torchvision.utils.save_image(samples.clamp_(0.0, 1.0), os.path.join(this_sample_dir, '%d.png'%r), nrow=10, normalize=False)
        
        
