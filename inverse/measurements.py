'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel
import numpy as np
from resizer import Resizer
from img_utils import Blurkernel, fft2_m
import math 
import os

from argparse import Namespace
import numpy as np
import scico
from scico import linop
import torch
import os


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)
    
def get_kspace(img, axes, im_sz):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(img, dim=axes), dim=axes), dim=axes) / im_sz

def kspace_to_image(kspace, axes, im_sz):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(kspace, dim=axes), dim=axes), dim=axes) * im_sz


@register_operator(name='mri')
class kspace_downsampler(LinearOperator):
    # preform 4 x acceleration --> keep 8% center, and randomly sample 17% other place
    def __init__(self, in_shape, mode='equispace', device='cuda'):
        self.im_sz = in_shape
        self.mode = mode
        self.mask = self.mask_gen()
        self.device = device

    def mask_gen(self):
        if self.mode == 'equispace':
            p_center = 0.08
            p_all = 0.25
            center = math.ceil(self.im_sz / 2)
            center_chosen = int(p_center * self.im_sz // 2)
            center_left =  center - center_chosen
            center_right = center + center_chosen
            rest_chosen = int(p_all * self.im_sz) - (center_right - center_left)
            step = math.ceil(center_left * 2 / rest_chosen)
            mask = torch.zeros((1, 1, self.im_sz, self.im_sz))
            mask[:, :, center_left : center_right, :] = 1.0
            left_chosen = np.arange(0, center_left, step)
            right_chosen = np.arange(self.im_sz - 1, center_right - 1, -step)
            mask[:, :, left_chosen] = 1.0
            mask[:, :, right_chosen] = 1.0
            print("keep {} of kspace".format(mask.sum().item() / self.im_sz /self.im_sz))
            return mask
        else:
            raise NotImplementedError

    def forward(self, img,  return_img=False, **kwargs):
        mask = self.mask.to(self.device)
        b, c, h, w = img.shape
        assert h == self.im_sz and w == self.im_sz and c == 1
        assert mask.shape[1:] == img.shape[1:]
        axes = (-2, -1)
        img_normalize = img
        #img_normalize = (img - img.min().item()) / (img.max().item() - img.min().item())
        img_kspace = get_kspace(img=img_normalize, axes=axes, im_sz=self.im_sz)
        img_kspace_downsampled = img_kspace * mask
        if return_img:
            img_normalize = (img - img.min().item()) / (img.max().item() - img.min().item())
            img_kspace_norm = torch.log(torch.abs(img_kspace))
            img_kspace_norm = (img_kspace_norm - img_kspace_norm.min()) / (img_kspace_norm.max() - img_kspace_norm.min())
            img_kspace_downsampled_norm = img_kspace_norm * mask
            kspace_input = img_kspace_downsampled.real + img_kspace_downsampled.imag * 1j
            img_recons = kspace_to_image(kspace=kspace_input, axes=axes, im_sz=self.im_sz).real
            img_recons = (img_recons - img_recons.min()) / (img_recons.max() - img_recons.min())
            all_images = torch.stack([img_normalize, img_kspace_norm, img_recons, img_kspace_downsampled_norm], dim=1)
            all_images = all_images.clone().detach().cpu() # b, 4, 1, im_sz, im_sz
            return img_kspace_downsampled, all_images
        
        return img_kspace_downsampled 
    
    def transpose(self, data, **kwargs):
        return data

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        # set random seed
        torch.manual_seed(0)
        np.random.seed(0)

        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        # amplitude = fft2_m(padded)**2
        return amplitude
    
    def transpose(self, data):
        # TODO: what should be the transpose of this operator?
        return data

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

    def transpose(self, data):
        return data
    


def get_cs_A(img_size, n_measurements, operator_path=None, init=True):
    """Get compressed sensing operator A."""
    if init:
        return CSOperatorScico(img_size, n_measurements)
    else:
        return CSOperatorScico.from_file(operator_path)


class CSOperatorScico(linop.LinearOperator):
    """Scico wrapper for compressed sensing operator."""

    def __init__(self, img_size, n_measurements):
        hparams = Namespace()
        hparams.n_measurements = n_measurements
        hparams.img_size = img_size
        hparams.n_input = np.prod(img_size)
        hparams.A = np.random.randn(hparams.n_input)
        hparams.sign_pattern = np.float32(
            (np.random.rand(hparams.n_input) < 0.5) * 2 - 1.0
        )
        hparams.indices = np.random.choice(
            hparams.n_input, hparams.n_measurements, replace=False
        )

        self.hparams = hparams
        self.op = CSOperatorTorch(hparams)

        self._init_scico()

    def _init_scico(self):
        fwd_op_numpy = lambda x: self.op(torch.from_numpy(np.array(x))).cpu().numpy()
        adjoint_op_numpy = (
            lambda x: self.op.adjoint(torch.from_numpy(np.array(x))).cpu().numpy()
        )

        super().__init__(
            input_shape=(self.hparams.img_size),
            output_shape=(self.hparams.n_measurements,),
            eval_fn=fwd_op_numpy,
            adj_fn=adjoint_op_numpy,
            jit=False,
        )

    def save_operator(self, fn):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        np.save(fn, self.hparams)

    def load_operator(self, fn):
        hparams = np.load(fn, allow_pickle=True).item()
        self.hparams = hparams
        self.op = CSOperatorTorch(hparams)
        self._init_scico()

    @classmethod
    def from_file(cls, fn):
        hparams = np.load(fn, allow_pickle=True).item()
        ins = cls(hparams.img_size, hparams.n_measurements)
        ins.load_operator(fn)
        return ins

    def get_norm(self):
        return self.op.norm()

    def get_ls(self, y):
        return self.op.ls(torch.tensor(y)).numpy()

@register_operator(name='cs')
class CSOperatorTorch(torch.nn.Module):
    """Torch wrapper for compressed sensing operator."""

    def __init__(self, n_measurements, n_input, A, sign_pattern, indices, device='cuda'):
        """
        Inputs:
            hparams: Namespace
                A: (n_input)
                sign_pattern: (n_input)
                indices: (n_measurements)
                n_measurements: int
                n_input: int
        """
        super().__init__()
        self.n_measurements =  n_measurements
        self.n_input =  n_input

        # to torch
        A = torch.from_numpy( A).float().to(device)
        sign_pattern = torch.from_numpy( sign_pattern).float().to(device)
        indices = torch.from_numpy( indices).long().to(device)
        self.indices = indices

        # register buffers
        self.register_buffer("A", A)
        self.register_buffer("sign_pattern", sign_pattern)

        # input output size
        self.input_size = (int(np.sqrt(self.n_input)), int(np.sqrt(self.n_input)))
        self.output_size = (self.n_measurements,)

        self.name = 'cs'
 

    def forward(self, x, **kwargs):
        """Inputs:
            x: (sqrt(n_input), sqrt(n_input)), torch tensor
        Outputs:
            y: (n_measurements)
        """
        x = x.reshape((self.n_input))
        z = _ifft(_fft(x * self.sign_pattern) * _fft(self.A))
        y = z[self.indices]
        y = torch.real(y)
        return y
    
    

    def adjoint(self, y):
        """Inputs:
            y: (n_measurements), torch tensor
        Outputs:
            x: (sqrt(n_input), sqrt(n_input))
        """
        z = torch.zeros(self.n_input, dtype=torch.float, device=y.device)
        z[self.indices] = y
        x = _ifft(_fft(z) * torch.conj(_fft(self.A)))
        x = x * self.sign_pattern
        x = x.reshape(self.input_size)
        x = torch.real(x)
        return x
    
    def A_adjoint(self, x):
        return self.adjoint(x)

    def adjoint_vjp(self, y):
        """Inputs:
            y: (n_measurements), torch tensor
        Outputs:
            x: (sqrt(n_input), sqrt(n_input))
        """
        inputs = torch.zeros(self.input_size, dtype=torch.complex64)
        _, vjp = torch.autograd.functional.vjp(self.forward, inputs, y)
        vjp = torch.real(vjp)
        return vjp

    def pinv(self, y):
        """Inputs:
            y: (n_measurements), torch tensor
        Outputs:
            x: (sqrt(n_input), sqrt(n_input))
        """
        z = torch.zeros(self.n_input).to(y.device)
        z[self.indices] = y
        x = _ifft(_fft(z) / _fft(self.A))
        x = x / self.sign_pattern
        x = x.reshape(self.input_size)
        x = torch.real(x)
        return x

    def ls(self, y, d=1e-6, verbose=True):
        """
        Least square solution
        Inputs:
            y: (n_measurements), torch tensor
            d: damping factor, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html#scipy.sparse.linalg.lsqr
        Outputs:
            x: (sqrt(n_input), sqrt(n_input))
        """
        # use gradient descent
        x = torch.zeros(self.input_size, dtype=torch.float, device=y.device)
        eta = 1e-1
        rel_change = []
        for _ in range(1000):
            x_old = x.clone()
            x = x - eta * (self.adjoint(self.forward(x) - y) - x * d)
            rel_change.append((torch.norm(x - x_old) / torch.norm(x)).item())
            print(rel_change[-1])
            if rel_change[-1] < 1e-10:
                if verbose:
                    print("ls rel change converged!")
                break
        else:
            if verbose:
                print("Warning: ls rel change not converged!")
        return x

    def gram(self, x):
        """
        Inputs:
            x: (sqrt(n_input), sqrt(n_input)), torch tensor
        Outputs:
            y: (n_measurements)
        """
        return self.adjoint(self.forward(x))

    def norm(self, verbose=False):
        """
        Find operator norm by power method.
        (the largest singular value of the measurement matrix)
        """
        x = torch.randn(self.input_size, dtype=torch.float)
        for i in range(1000):
            x = self.gram(x)
            x = x / torch.norm(x)

            # Rayleigh quotient (largest eigenvalue of the gram matrix)
            R = torch.sum(self.gram(x) * x) / (torch.norm(x) ** 2)

            # operator norm (square root of largest eigenvalue)
            n = torch.sqrt(R)

            if i > 0 and torch.abs(n - n_old) < 1e-2:
                if verbose:
                    print("norm converged!")
                break

            n_old = n
        else:
            if verbose:
                print("Warning: norm not converged!")
        return n.item()
    
    def transpose(self, data):
        return self.adjoint(data)
    

@register_operator(name='cs1')
class CSOperatorTorch1(torch.nn.Module):
    """Torch wrapper for compressed sensing operator."""

    def __init__(self, n_measurements, n_input, A, sign_pattern, indices, device='cuda'):
        """
        Inputs:
            hparams: Namespace
                A: (n_input)
                sign_pattern: (n_input)
                indices: (n_measurements)
                n_measurements: int
                n_input: int
        """
        super().__init__()
        self.n_measurements =  n_measurements
        self.n_input =  n_input

        # to torch
        A = torch.from_numpy( A).float().to(device)
        sign_pattern = torch.from_numpy( sign_pattern).float().to(device)
        indices = torch.from_numpy( indices).long().to(device)
        self.indices = indices

        # register buffers
        self.register_buffer("Am", A)
        self.register_buffer("sign_pattern", sign_pattern)

        # input output size
        self.input_size = (int(np.sqrt(self.n_input)), int(np.sqrt(self.n_input)))
        self.output_size = (self.n_measurements,)

        self.name = 'cs'
 

    def forward(self, x, **kwargs):
        """Inputs:
            x: (sqrt(n_input), sqrt(n_input)), torch tensor
        Outputs:
            y: (n_measurements)
        """
        x = x.reshape((self.n_input))
        z = _ifft(_fft(x * self.sign_pattern) * _fft(self.Am))
        z[self.indices] = 0
        y = torch.real(z)
        y = y.reshape(1, 1, *self.input_size)
        return y
    
    def A(self, x):
        return self.forward(x)
    
    

    def adjoint(self, y):
        """Inputs:
            y: (n_measurements), torch tensor
        Outputs:
            x: (sqrt(n_input), sqrt(n_input))
        """
         
        x = _ifft(_fft(y.reshape(self.n_input)) * torch.conj(_fft(self.Am)))
        x = x * self.sign_pattern
        x = x.reshape(1,1,*self.input_size)
        x = torch.real(x)
        return x
    
    def A_adjoint(self, x):
        return self.adjoint(x)

    def adjoint_vjp(self, y):
        """Inputs:
            y: (n_measurements), torch tensor
        Outputs:
            x: (sqrt(n_input), sqrt(n_input))
        """
        inputs = torch.zeros(1,1, *self.input_size, dtype=torch.complex64, device=y.device)
        _, vjp = torch.autograd.functional.vjp(self.forward, inputs, y)
        vjp = torch.real(vjp)
        return vjp
    
    def A_vjp(self, v, x):
        return self.adjoint_vjp(x)

    def pinv(self, y):
        """Inputs:
            y: (n_measurements), torch tensor
        Outputs:
            x: (sqrt(n_input), sqrt(n_input))
        """
        z = torch.zeros(self.n_input).to(y.device)
        z[self.indices] = y
        x = _ifft(_fft(z) / _fft(self.Am))
        x = x / self.sign_pattern
        x = x.reshape(self.input_size)
        x = torch.real(x)
        return x

    def ls(self, y, d=1e-6, verbose=True):
        """
        Least square solution
        Inputs:
            y: (n_measurements), torch tensor
            d: damping factor, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html#scipy.sparse.linalg.lsqr
        Outputs:
            x: (sqrt(n_input), sqrt(n_input))
        """
        # use gradient descent
        x = torch.zeros(self.input_size, dtype=torch.float, device=y.device)
        eta = 1e-1
        rel_change = []
        for _ in range(1000):
            x_old = x.clone()
            x = x - eta * (self.adjoint(self.forward(x) - y) - x * d)
            rel_change.append((torch.norm(x - x_old) / torch.norm(x)).item())
            print(rel_change[-1])
            if rel_change[-1] < 1e-10:
                if verbose:
                    print("ls rel change converged!")
                break
        else:
            if verbose:
                print("Warning: ls rel change not converged!")
        return x

    def gram(self, x):
        """
        Inputs:
            x: (sqrt(n_input), sqrt(n_input)), torch tensor
        Outputs:
            y: (n_measurements)
        """
        return self.adjoint(self.forward(x))

    def norm(self, verbose=False):
        """
        Find operator norm by power method.
        (the largest singular value of the measurement matrix)
        """
        x = torch.randn(self.input_size, dtype=torch.float)
        for i in range(1000):
            x = self.gram(x)
            x = x / torch.norm(x)

            # Rayleigh quotient (largest eigenvalue of the gram matrix)
            R = torch.sum(self.gram(x) * x) / (torch.norm(x) ** 2)

            # operator norm (square root of largest eigenvalue)
            n = torch.sqrt(R)

            if i > 0 and torch.abs(n - n_old) < 1e-2:
                if verbose:
                    print("norm converged!")
                break

            n_old = n
        else:
            if verbose:
                print("Warning: norm not converged!")
        return n.item()
    
    def transpose(self, data):
        return self.adjoint(data)


def _fft(x):
    return torch.fft.fft(x, norm="ortho")


def _ifft(x):
    return torch.fft.ifft(x, norm="ortho")
# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)






 



