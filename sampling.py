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
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils

import matplotlib.pyplot as plt
import logging

import torchvision
from tqdm import tqdm
# import partial function
from functools import partial


import os
from inverse.img_utils import clear_color
from matplotlib import pyplot as plt


def get_sampling_fn(config, sde, shape, inverse_scaler, eps, inverse=False):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'rectified_flow':
    if not inverse:
      sampling_fn = get_rectified_flow_sampler(sde=sde, shape=shape, inverse_scaler=inverse_scaler, device=config.device)
    else:
      sampling_fn = get_rectified_flow_inverse_sampler(sde=sde, shape=shape, inverse_scaler=inverse_scaler, device=config.device)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


def get_rectified_flow_sampler(sde, shape, inverse_scaler, device='cuda'):
  """
  Get rectified flow sampler

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  def euler_sampler(model, z=None):
    """The probability flow ODE sampler with simple Euler discretization.

    Args:
      model: A velocity model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
        x = z0.detach().clone()
      else:
        x = z
      
      model_fn = mutils.get_model_fn(model, train=False) 
      
      ### Uniform
      dt = 1./sde.sample_N
      eps = 1e-3 # default: 1e-3
      for i in range(sde.sample_N):
        
        num_t = i /sde.sample_N * (sde.T - eps) + eps
        t = torch.ones(shape[0], device=device) * num_t
        pred = model_fn(x, t*999) ### Copy from models/utils.py 

        # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability 
        sigma_t = sde.sigma_t(num_t)
        pred_sigma = pred + (sigma_t**2)/(2*(sde.noise_scale**2)*((1.-num_t)**2)) * (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*x.detach().clone())

        x = x.detach().clone() + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)
      
      x = inverse_scaler(x)
      nfe = sde.sample_N
      return x, nfe
  
  def rk45_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A velocity model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      rtol=atol=sde.ode_tol
      method='RK45'
      eps=1e-3

      # Initial sample
      if z is None:
        z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
        x = z0.detach().clone()
      else:
        x = z
      
      model_fn = mutils.get_model_fn(model, train=False)

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = model_fn(x, vec_t*999)

        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (eps, sde.T), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      x = inverse_scaler(x)
      
      return x, nfe
  

  print('Type of Sampler:', sde.use_ode_sampler)
  if sde.use_ode_sampler=='rk45':
      return rk45_sampler
  elif sde.use_ode_sampler=='euler':
      return euler_sampler
  else:
      assert False, 'Not Implemented!'

def get_rectified_flow_inverse_sampler(sde, shape, inverse_scaler,   device='cuda'):
  """
  Get rectified flow sampler

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  def euler_sampler(model, measurement, operator, z=None, init=0.0, record=False, beta = 1,  task = 'cs', lamda=1, num_iter=1, eta_scheduler = 'constant',  save_root=None, alpha=1, method='naive', eta=1, n_trace=1, zeta=1, nita=10, stop_time=1, **kwargs):
    """The probability flow ODE sampler with simple Euler discretization.

    Args:
      model: A velocity model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    
    # Initial sample
    if z is None:
      z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
      x = z0.detach().clone()
    else:
      x = z
    
    model_fn = mutils.get_model_fn(model, train=False) 
    
    ### Uniform
    dt = 1./sde.sample_N
    eps = 1e-3 # default: 1e-3

    if init > 0.0:
      eps = 1 * init
      dt = (1 - eps) / sde.sample_N
      
    errors = []
    errors1 = []
    errors2 = []
    errors3 = []
    norm_errors = []


    if method == 'ours':
      u0 = operator.forward(x, **kwargs)
      u = u0.detach().clone()
    elif method == 'ours_x' or method == 'ours_xx' or method == 'ours_adaptive' or method == 'ours_adaptive_k' or method == 'ours_xdirect' or 'ours_' in method:
      u0 = operator.forward(x, **kwargs)
 
      shape_eps = (n_trace, ) + x.shape[1:] 
      
  
    for i in tqdm(range(sde.sample_N)):
      
      num_t = i /sde.sample_N * (sde.T - eps) + eps
      t = torch.ones(shape[0], device=device) * num_t
 
      if method == 'ours_x' or method == 'ours_xx' or method == 'ours_adaptive' or method == 'ours_adaptive_k' or method == 'ours_xdirect' or 'ours_' in method:
        func = partial(model_fn, labels=t*999)
        div_fn = get_hutchinson_div_fn_torch(func)

 
      
      if method=='naive':
        x_ = x.detach().clone() 
        x_.requires_grad = True
       
        pred = model_fn(x_, t*999) ### Copy from models/utils.py 
        pred_sigma = pred 
        x1_hat = x_ + pred_sigma * (1-num_t) # one_step estimation
        if task == 'cs':
          y_pred = operator.forward(x1_hat[0][0], **kwargs)
          y_pred = y_pred.unsqueeze(0).unsqueeze(0)
        else:
          y_pred = operator.forward(x1_hat, **kwargs)
        error = torch.linalg.norm(y_pred - measurement) 
        
        grad = torch.autograd.grad(error, x_, create_graph=False)[0]

        #print('Error:', error)
        errors.append(error.item())

        if eta == 0:
          pred_sigma = pred_sigma -  grad
        else:
          pred_sigma = pred_sigma - eta * grad
        
        x = x + pred_sigma * dt 
        x = x.detach().clone()
      elif method=='fm':
        x_ = x.detach().clone() 
        x_.requires_grad = True
        pred = model_fn(x_, t*999) ### Copy from models/utils.py 
        pred_sigma = pred 
        x1_hat = x_ + pred_sigma * (1-num_t) # one_step estimation
        
        n = 1
        m = torch.prod(torch.tensor(measurement.shape[1:]))
        # m = 256*256*3
        sigmay = 0.01
        choice = 'noisy'
        if choice == 'noise_free':
          rt2 = (1- num_t)**2 / (num_t**2 + (1-num_t)**2)
       

          mat = (operator.pinv(measurement) - operator.pinv(operator.forward(x1_hat))).reshape(n, -1)
          mat_x = 1/rt2 *  (mat.detach() * x1_hat.reshape(n, -1)).sum()
          grad = torch.autograd.grad(mat_x, x_, retain_graph=False)[0]
          logging.info(f'Error:, {mat_x.item()}')
          if eta == 0:
            pred_sigma = pred_sigma  + (1-num_t)/num_t * grad
          else:
            pred_sigma = pred_sigma +  eta * (1-num_t)/num_t * grad
          
        
          x = x + pred_sigma * dt 
          x = x.detach().clone()
        elif choice == 'noisy':
          mat =  measurement.reshape(1, -1) - operator.H(x1_hat)
          # print('shape', mat.shape)
          rt2 = (1- num_t)**2 / (num_t**2 + (1-num_t)**2)
          save_memory= False
          if save_memory:
            # a = sigmay**2 * torch.diag(torch.ones(m, device=measurement.device))
            for kk in tqdm(range(m)):
              temp = torch.zeros((1,m), device=measurement.device)
              temp[0, kk] = 1
              A = rt2 * operator.H(operator.Ht(temp)) 
              A[0, kk] += sigmay**2
            
            # Cholesky decomposition
            L = torch.linalg.cholesky(A)
            inv_L = torch.inverse(L)
            mat1 = inv_L @ inv_L.T  
          else:
            mat1 = torch.linalg.inv( rt2*  operator.H(operator.Ht(torch.eye(m, device=measurement.device))) + sigmay**2 * torch.eye(m, device=measurement.device))
          mat_x = ( (mat @ mat1).detach()  * operator.H(x1_hat)).sum()
          grad = torch.autograd.grad(mat_x, x_, retain_graph=False)[0]
          logging.info(f'Error:, {mat_x.item()}')

          if eta == 0:
            pred_sigma = pred_sigma  + (1-num_t)/num_t * grad
          else:
            pred_sigma = pred_sigma +  eta * grad
          
          x = x + pred_sigma * dt 
          x = x.detach().clone()
      elif method == 'ours_x':
        k = 0
        
        if i != 0:
          initial_x = x.detach().clone()
        while True:
          x_ = x.detach().clone() 
          x_.requires_grad = True
        
          pred = model_fn(x_, t*999)
          xnext = x_ + pred * dt
          eps_ = draw_epsilon_torch(shape_eps, 'rademacher')
          tr_jac = div_fn(x_,  [eps_])
         
          tr_jac = tr_jac.mean()
          
          if i == 0:           
            # TODO: noise level is 0.01 right now but may change
            # lik = 1/(2* (num_t + dt)**2 * 0.01**2) * torch.linalg.norm(operator.forward(xnext)- ((num_t + dt)*measurement+(1-num_t-dt)*u0))**2
            lik = lamda * torch.linalg.norm(operator.forward(xnext)- ((num_t + dt)*measurement+(1-num_t-dt)*u0))**2 
            prior_lik =  1/2 * torch.linalg.norm(x_)**2
            prior_change = tr_jac * dt
             
            # print('Likelihood:', lik.item(), end=' ')
            # print('Prior Likelihood:', prior_lik.item(), end=' ')
            # print('Prior Change:', prior_change.item())    
            

           
          else:
            # lik = 1/(2* (num_t + dt)**2 * 0.01**2) * torch.linalg.norm(operator.forward(xnext)- ((num_t + dt)*measurement+(1-num_t-dt)*u0))**2
            lik = lamda * torch.linalg.norm(operator.forward(xnext)- ((num_t + dt)*measurement+(1-num_t-dt)*u0))**2
            prior_lik =  1/2 * torch.linalg.norm(x_ - initial_x)**2
            prior_change = tr_jac * dt
            
            # print('Likelihood:', lik.item(), end=' ')
            # print('Prior Likelihood:', prior_lik.item(), end=' ')
            # print('Prior Change:', prior_change.item())
          
           
          loss = lik + prior_lik + prior_change * zeta 
           
          # change print to logging.info
          logging.info(f'Log likelihood:, {lik.item()}')
          logging.info(f'Prior likelihood:, {prior_lik.item()}')
          logging.info(f'Prior change:, {prior_change.item()}')
          
          errors1.append(lik.item())
          errors2.append(prior_lik.item())
          errors3.append(prior_change.item())

      
          logging.info(f'Loss:, {loss.item()}')
          errors.append(loss.item())
          grad = torch.autograd.grad(loss, x_, create_graph=False)[0]
          x = x - eta * grad
          k += 1
          if k == num_iter:
            break

        vt = model_fn(x, t*999)
        x = x + vt * dt      
        x = x.detach().clone()
      
      elif method == 'ours_xx':
        k = 0

        # use Adam optimizer to update x
        x_ = x.detach().clone() 
        x_.requires_grad = True
        optimizer = torch.optim.Adam([x_], lr=eta)
        
     
        while True:
          if i >= sde.sample_N*stop_time:
            break

          optimizer.zero_grad()
        
          pred = model_fn(x_, t*999)
          xnext = x_ + pred * dt
          eps_ = draw_epsilon_torch(shape_eps, 'rademacher')
          tr_jac = div_fn(x_,  [eps_])
          tr_jac = tr_jac.mean()
          if lamda == 0:
            lik = 1/(2* (num_t + dt)**2 * 0.01**2) * torch.linalg.norm(operator.forward(xnext, **kwargs)- ((num_t + dt)*measurement+(1-num_t-dt)*u0)) **2
          else:
            lik = lamda    * torch.linalg.norm(operator.forward(xnext, **kwargs)- ((num_t + dt)*measurement+(1-num_t-dt)*u0))**2
          prior_change = tr_jac * dt

          loss = lik +   prior_change * zeta 
          if i == 0:
            loss = loss + 1/2 * torch.linalg.norm(x_)**2
           
          # change print to logging.info
          logging.info(f'Log likelihood:, {lik.item()}')
          logging.info(f'Prior change:, {prior_change.item()}')


          norm_error = torch.linalg.norm(measurement - operator.forward(x_ + pred * (1- num_t)), **kwargs)
          norm_errors.append(norm_error.item())
          logging.info(f'Norm Error:, {norm_error.item()}')
          
          errors1.append(lik.item())
      
          errors3.append(prior_change.item())

      
          logging.info(f'Loss:, {loss.item()}')
          errors.append(loss.item())
          # grad = torch.autograd.grad(loss, x_, create_graph=False)[0]
          loss.backward()
          if i == 0:
            grad_xt_lik = 0
          else:
            grad_xt_lik = - nita / (1-num_t)   * (-x_ + num_t * pred) 
          
          x_.grad = x_.grad + grad_xt_lik
          # x = x - eta *   (grad + grad_xt_lik) # some scheduler... 
          # x = x - eta *    (grad + grad_xt_lik) # some scheduler... 

          optimizer.step()

          k += 1
          if k == num_iter:
            break

        x = x_
        
        vt = model_fn(x, t*999)
        x = x + vt * dt      
        x = x.detach().clone()
      elif method == 'ours_xx_noise':
        k = 0


     
        while True:
          if i >= sde.sample_N*stop_time:
            break

          x_ = x.detach().clone() 
          x_.requires_grad = True

          
        
          pred = model_fn(x_, t*999)
          xnext = x_ + pred * dt
          eps_ = draw_epsilon_torch(shape_eps, 'rademacher')
          tr_jac = div_fn(x_,  [eps_])
          tr_jac = tr_jac.mean()
          if lamda == 0:
            lik = 1/(2* (num_t + dt)**2 * 0.01**2) * torch.linalg.norm(operator.forward(xnext)- ((num_t + dt)*measurement+(1-num_t-dt)*u0)) **2
          else:
            lik = lamda    * torch.linalg.norm(operator.forward(xnext)- ((num_t + dt)*measurement+(1-num_t-dt)*u0))**2
          prior_change = tr_jac * dt

          loss = lik +   prior_change * zeta 
          if i == 0:
            loss = loss + 1/2 * torch.linalg.norm(x_)**2
           
          # change print to logging.info
          logging.info(f'Log likelihood:, {lik.item()}')
          logging.info(f'Prior change:, {prior_change.item()}')


          norm_error = torch.linalg.norm(measurement - operator.forward(x_ + pred * (1- num_t)))
          norm_errors.append(norm_error.item())
          logging.info(f'Norm Error:, {norm_error.item()}')
          
          errors1.append(lik.item())
      
          errors3.append(prior_change.item())

      
          logging.info(f'Loss:, {loss.item()}')
          errors.append(loss.item())
          grad = torch.autograd.grad(loss, x_, create_graph=False)[0]
          # loss.backward()
          if i == 0:
            grad_xt_lik = 0
          else:
            grad_xt_lik = - nita / (1-num_t)   * (-x_ + num_t * pred) 
          
       
          # x = x - eta *   (grad + grad_xt_lik) # some scheduler... 
          x_ = x_ - eta *    (grad + grad_xt_lik) + beta* (2* eta)** (1/2)* torch.randn_like(x_) # some scheduler... 

           

          k += 1
          if k == num_iter:
            break

        x = x_
        
        vt = model_fn(x, t*999)
        x = x + vt * dt      
        x = x.detach().clone()

      elif method == 'ours_xx_with_l2':
        k = 0

        x_ = x.detach().clone() 
    
        while True:
          if i >= sde.sample_N*stop_time:
            break

          # use Adam optimizer to update x
          x_ = x_.detach().clone()
          x_.requires_grad = True
          if eta_scheduler == '-1':
            optimizer = torch.optim.Adam([x_], lr=eta*(k+1)**(-2))
          elif eta_scheduler == '-2':
            optimizer = torch.optim.Adam([x_], lr=eta*(k+1)**(-2))
          elif eta_scheduler == 'constant':
            optimizer = torch.optim.Adam([x_], lr=eta)
            # use gradient descent to update x
            # optimizer = torch.optim.SGD([x_], lr=eta * 0.5 ** (sde.sample_N - i + 1))
        

          optimizer.zero_grad()
        
          pred = model_fn(x_, t*999)
          xnext = x_ + pred * dt
          eps_ = draw_epsilon_torch(shape_eps, 'rademacher')
          tr_jac = div_fn(x_,  [eps_])
          tr_jac = tr_jac.mean()
          if lamda == 0:
            lik = 1/(2* (num_t + dt)**2 * 0.01**2) * torch.linalg.norm(operator.forward(xnext, **kwargs)- ((num_t + dt)*measurement+(1-num_t-dt)*u0)) **2
          else:
            lik = lamda    * torch.linalg.norm(operator.forward(xnext, **kwargs)- ((num_t + dt)*measurement+(1-num_t-dt)*u0)) **2
          prior_change = tr_jac * dt

          
          loss = lik +   prior_change * zeta  
          if i == 0:
            loss = loss + 1/2 * torch.linalg.norm(x_)**2
           
          # change print to logging.info
          logging.info(f'Log likelihood:, {lik.item()}')
          logging.info(f'Prior change:, {prior_change.item()}')


          norm_error = torch.linalg.norm(measurement - operator.forward(x_ + pred * (1- num_t), **kwargs))
          norm_errors.append(norm_error.item())
          logging.info(f'Norm Error:, {norm_error.item()}')
          
          errors1.append(lik.item())
      
          errors3.append(prior_change.item())

      
          logging.info(f'Loss:, {loss.item()}')
          errors.append(loss.item())
          grad = torch.autograd.grad(loss, x_, create_graph=False)[0]
          # loss.backward(retain_graph=False)
          if i == 0:
            grad_xt_lik = 0
          else:
            grad_xt_lik = - nita / (1-num_t)   * (-x_ + num_t * pred) 
          
          x_.grad = grad +  grad_xt_lik 
          # x = x - eta *   (grad + grad_xt_lik) # some scheduler... 
          # x = x - eta *    (grad + grad_xt_lik) # some scheduler... 

          

          optimizer.step()

          x_ = x_ +  beta * torch.randn_like(x_)
          

          
          k += 1
          if k == num_iter:
            break

        x = x_
        
        vt = model_fn(x, t*999)
        x = x + vt * dt
        x = x.detach().clone()
      
      elif method == 'ours_xx_with_l2_gaussian':
        k = 0

        x_ = x.detach().clone() 
    
        while True:
          if i >= sde.sample_N*stop_time:
            break

          # use Adam optimizer to update x
          x_ = x_.detach().clone()
          x_.requires_grad = True
          optimizer = torch.optim.Adam([x_], lr=eta)
        

          optimizer.zero_grad()
        
          pred = model_fn(x_, t*999)
          xnext = x_ + pred * dt
          eps_ = draw_epsilon_torch(shape_eps, 'rademacher')
          tr_jac = div_fn(x_,  [eps_])
          tr_jac = tr_jac.mean()
          if lamda == 0:
            lik = 1/(2* (num_t + dt)**2 * 0.01**2) * torch.linalg.norm(operator.forward(xnext, **kwargs)- ((num_t + dt)*measurement+(1-num_t-dt)*u0)) **2
          else:
            lik = lamda    * torch.linalg.norm(operator.forward(xnext, **kwargs)- ((num_t + dt)*measurement+(1-num_t-dt)*u0)) **2
          prior_change = tr_jac * dt

          
          loss = lik +   prior_change * zeta  
          if i == 0:
            loss = loss + 1/2 * torch.linalg.norm(x_)**2
           
          # change print to logging.info
          logging.info(f'Log likelihood:, {lik.item()}')
          logging.info(f'Prior change:, {prior_change.item()}')


          norm_error = torch.linalg.norm(measurement - operator.forward(x_ + pred * (1- num_t), **kwargs))
          norm_errors.append(norm_error.item())
          logging.info(f'Norm Error:, {norm_error.item()}')
          
          errors1.append(lik.item())
      
          errors3.append(prior_change.item())

      
          logging.info(f'Loss:, {loss.item()}')
          errors.append(loss.item())
          grad = torch.autograd.grad(loss, x_, create_graph=False)[0]
          # loss.backward()
          if i == 0:
            grad_xt_lik = 0
          else:
            grad_xt_lik = - nita / (1-num_t)   * (-x_ + num_t * pred) 
          
          x_.grad = grad + grad_xt_lik 
          # x = x - eta *   (grad + grad_xt_lik) # some scheduler... 
          # x = x - eta *    (grad + grad_xt_lik) # some scheduler... 

          

          optimizer.step()

          x_ = x_ +  beta * torch.randn_like(x_)
          

          
          k += 1
          if k == num_iter:
            break

        x = x_
        
        vt = model_fn(x, t*999)
        x = x + vt * dt
        x = x.detach().clone()


      elif method == 'ours_xx_with_prior':
        k = 0
        xhat = x + model_fn(x, t*999) * dt

        # use Adam optimizer to update x
        x_ = x.detach().clone() 
        x_.requires_grad = True
        optimizer = torch.optim.Adam([x_], lr=eta)
        
     
        while True:
          if i >= sde.sample_N*stop_time:
            break

          optimizer.zero_grad()
        
          pred = model_fn(x_, t*999)
          xnext = x_ + pred * dt
          eps_ = draw_epsilon_torch(shape_eps, 'rademacher')
          tr_jac = div_fn(x_,  [eps_])
          tr_jac = tr_jac.mean()
          if lamda == 0:
            lik = 1/(2* (num_t + dt)**2 * 0.01**2) * torch.linalg.norm(operator.forward(xnext)- ((num_t + dt)*measurement+(1-num_t-dt)*u0)) **2
          else:
            lik = lamda    * torch.linalg.norm(operator.forward(xnext)- ((num_t + dt)*measurement+(1-num_t-dt)*u0))**2
          prior_change = tr_jac * dt

          loss = lik +   prior_change * zeta 
          if i == 0:
            loss = loss + 1/2 * torch.linalg.norm(x_)**2
           
          # change print to logging.info
          logging.info(f'Log likelihood:, {lik.item()}')
          logging.info(f'Prior change:, {prior_change.item()}')


          norm_error = torch.linalg.norm(measurement - operator.forward(x_ + pred * (1- num_t)))
          norm_errors.append(norm_error.item())
          logging.info(f'Norm Error:, {norm_error.item()}')
          
          errors1.append(lik.item())
      
          errors3.append(prior_change.item())

      
          logging.info(f'Loss:, {loss.item()}')
          errors.append(loss.item())
          # grad = torch.autograd.grad(loss, x_, create_graph=False)[0]
          loss.backward()
          if i == 0:
            grad_xt_lik = 0
          else:
            grad_xt_lik = - nita / (1-num_t)   * (-x_ + num_t * pred) 
          
          x_.grad = x_.grad + grad_xt_lik
          # x = x - eta *   (grad + grad_xt_lik) # some scheduler... 
          # x = x - eta *    (grad + grad_xt_lik) # some scheduler... 

          optimizer.step()

          k += 1
          if k == num_iter:
            break

        x = x_
        
        vt = model_fn(x, t*999)
        x = x + vt * dt      
         
        x = alpha * x + (1-alpha) * xhat
        x = x.detach().clone()
      elif method == 'ours_xdirect':
        k = 0

        # use Adam optimizer to update x
        x_ = x.detach().clone() 
        x_.requires_grad = True
        optimizer = torch.optim.Adam([x_], lr=eta)
        
     
        while True:
          if i >= sde.sample_N*stop_time:
            break

          optimizer.zero_grad()
        
          pred = model_fn(x_, t*999)
          xnext = x_ + pred * 0
          # eps_ = draw_epsilon_torch(shape_eps, 'rademacher')
          # tr_jac = div_fn(x_,  [eps_])
          # tr_jac = tr_jac.mean()
          if lamda == 0:
            lik = 1/(2* (num_t + dt)**2 * 0.01**2) * torch.linalg.norm(operator.forward(xnext)- ((num_t  )*measurement+(1-num_t )*u0)) **2
          else:
            lik = lamda    * torch.linalg.norm(operator.forward(xnext)- ((num_t  )*measurement+(1-num_t  )*u0))**2
          
          loss = lik 
          if i == 0:
            loss = loss + 1/2 * torch.linalg.norm(x_)**2
           
          # change print to logging.info
          logging.info(f'Log likelihood:, {lik.item()}')
         

          norm_error = torch.linalg.norm(measurement - operator.forward(x_ + pred * (1- num_t)))
          norm_errors.append(norm_error.item())
          logging.info(f'Norm Error:, {norm_error.item()}')
          
          errors1.append(lik.item())
       
      
          logging.info(f'Loss:, {loss.item()}')
          errors.append(loss.item())
          # grad = torch.autograd.grad(loss, x_, create_graph=False)[0]
          loss.backward()
          if i == 0:
            grad_xt_lik = 0
          else:
            grad_xt_lik = - nita / (1-num_t)   * (-x_ + num_t * pred) 
          
          x_.grad = x_.grad + grad_xt_lik
          # x = x - eta *   (grad + grad_xt_lik) # some scheduler... 
          # x = x - eta *    (grad + grad_xt_lik) # some scheduler... 

          optimizer.step()

          k += 1
          if k == num_iter:
            break

        x = x_
        
        vt = model_fn(x, t*999)
        x = x + vt * dt      
        x = x.detach().clone()
      elif method == 'ours_adaptive_k':
        k = 0

        u0 = (operator.forward(x.clone(), **kwargs) - num_t * measurement) / (1 - num_t)
        
     
        while True:
          x_ = x.detach().clone() 
          x_.requires_grad = True
        
          pred = model_fn(x_, t*999)
          xnext = x_ + pred * dt
          eps_ = draw_epsilon_torch(shape_eps, 'rademacher')
          tr_jac = div_fn(x_,  [eps_])
          tr_jac = tr_jac.mean()
          if lamda == 0:
            lik = 1/(2* (num_t + dt)**2 * 0.01**2) * torch.linalg.norm(operator.forward(xnext)- ((num_t + dt)*measurement+(1-num_t-dt)*u0))**2 
          else:
            lik = lamda * (1-num_t)     *  torch.linalg.norm(operator.forward(xnext)- ((num_t + dt)*measurement+(1-num_t-dt)*u0)) **2
          prior_change = tr_jac * dt

          loss = lik +   prior_change * zeta 
          if i == 0:
            loss = loss + 1/2 * torch.linalg.norm(x_)**2
          
          # change print to logging.info
          logging.info(f'Log likelihood:, {lik.item()}')
          logging.info(f'Prior change:, {prior_change.item()}')
          
          errors1.append(lik.item())
      
          errors3.append(prior_change.item())

          norm_error = torch.linalg.norm(measurement - operator.forward(x_ + pred * (1- num_t)))
          norm_errors.append(norm_error.item())
          logging.info(f'Norm Error:, {norm_error.item()}')

      
          logging.info(f'Loss:, {loss.item()}')
          errors.append(loss.item())
          grad = torch.autograd.grad(loss, x_, create_graph=False)[0]
          if i == 0:
            grad_xt_lik = 0
          else:
            grad_xt_lik = - nita * (1-num_t)*   (-x_ + num_t *  pred) 
            # grad_xt_lik = - nita * pred
          x = x - eta * (1+k)**(-1) *  (grad + grad_xt_lik)
          k += 1
          if k == num_iter:
            break
        

        vt = model_fn(x, t*999)
        x = x + vt * dt      
        x = x.detach().clone()


      else:
        raise ValueError('Not Implemented!')

      

   
      if record:
        plt.imsave(os.path.join(save_root, 'progress',  f'{i}.png'), clear_color(inverse_scaler(x)))  

    # plot errors and save it
    plt.figure()
    plt.plot(errors)
    plt.savefig(os.path.join(save_root,  'errors.png'))
      
    if method == 'ours_x':
      plt.figure()
      plt.plot(errors1)
      plt.savefig(os.path.join(save_root,  'lik.png'))
      plt.figure()
      plt.plot(errors2)
      plt.savefig(os.path.join(save_root,  'prior_lik.png'))
      plt.figure()
      plt.plot(errors3)
      plt.savefig(os.path.join(save_root,  'prior_change.png'))
    elif method == 'ours_xx' or method == 'ours_adaptive' or 'ours_' in method:
      plt.figure()
      plt.plot(errors1)
      plt.savefig(os.path.join(save_root,  'lik.png'))
      plt.figure()
      plt.plot(norm_errors)
      plt.savefig(os.path.join(save_root,  'norm_error.png'))
      
      plt.figure()
      plt.plot(errors3)
      plt.savefig(os.path.join(save_root,  'prior_change.png'))
    x = inverse_scaler(x)
    nfe = sde.sample_N
    return x, nfe
  
  def rk45_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A velocity model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      rtol=atol=sde.ode_tol
      method='RK45'
      eps=1e-3

      # Initial sample
      if z is None:
        z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
        x = z0.detach().clone()
      else:
        x = z
      
      model_fn = mutils.get_model_fn(model, train=False)

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = model_fn(x, vec_t*999)

        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (eps, sde.T), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      x = inverse_scaler(x)
      
      return x, nfe
  

  print('Type of Sampler:', sde.use_ode_sampler)
  if sde.use_ode_sampler=='rk45':
    return rk45_sampler
  elif sde.use_ode_sampler=='euler':
    return euler_sampler
  else:
    assert False, 'Not Implemented!'


 

# write draw_epsilon function using pytorch
def draw_epsilon_torch(shape_,
                 hutchinson_type = 'rademacher'):
  """Draw an epsilon for Hutchinson-Skilling trace estimation
  Args:
    shape: A sequence of integers representing the expected shape of a single sample.
    hutchinson_type: A `str` representing the type of Hutchinson trace estimator.
  Returns:
    epsilon: A `torch.Tensor` of shape `shape` representing the Hutchinson trace estimator.
  """
  if hutchinson_type.lower() == 'gaussian':
    epsilon = torch.randn(shape_).to('cuda')
  elif hutchinson_type.lower() == 'rademacher':
    epsilon = 2 * (torch.randint(0, 2, shape_).float() - 0.5).to('cuda')
  else:
    raise ValueError(f'Hutchinson type {hutchinson_type} unknown')
  return epsilon

 


def get_hutchinson_div_fn_torch(fn):
  """Return the divergence function for `fn(x, t)`.
  Assumes that `fn` takes mini-batched inputs. The returned divergence function
  takes a batch of epsilons and returns a batch of trace estimators.
  Args:
    fn: The function to take the divergence of.
  Returns:
    div_fn: The divergence function that takes a batch of `epsilon`s
      for trace estimation.
  """

  def div_fn_one_trace_estimate(x,  epsilon):
    grad_fn = lambda data: torch.sum(fn(data ) * epsilon)
    tmp = grad_fn(x)
    grad_fn_eps = torch.autograd.grad(tmp, x, create_graph=False)[0]
    div = torch.sum(grad_fn_eps * epsilon, dim=tuple(range(1, len(x.shape))))
    return div

  def div_fn(x,   epsilons):
    return torch.mean(torch.stack([div_fn_one_trace_estimate(x,   epsilon) for epsilon in epsilons]), dim=0)

  return div_fn