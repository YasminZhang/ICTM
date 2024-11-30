# 🔥 [NeurIPS 2024] Flow Priors for Linear Inverse Problems via Iterative Corrupted Trajectory Matching
[![Static Badge](https://img.shields.io/badge/NeurIPS_2024_paper-arxiv_link-blue)
](https://arxiv.org/abs/2405.18816) 
[![Static Badge](https://img.shields.io/badge/NeurIPS_2024_paper-Open_Review-red)
](https://openreview.net/forum?id=1H2e7USI09) 



 

This repository hosts the code and resources associated with our paper  on utlizing flow priors to solve linear inverse problems.
 
 

## Abstract
 Generative models based on flow matching have attracted significant attention for their simplicity and superior performance in high-resolution image synthesis. By leveraging the instantaneous change-of-variables formula, one can directly compute image likelihoods from a learned flow, making them enticing candidates as priors for downstream tasks such as inverse problems. In particular, a natural approach would be to incorporate such image probabilities in a maximum-a-posteriori (MAP) estimation problem. A major obstacle, however, lies in the slow computation of the log-likelihood, as it requires backpropagating through an ODE solver, which can be prohibitively slow for high-dimensional problems. In this work, we propose an iterative algorithm to approximate the MAP estimator efficiently to solve a variety of linear inverse problems. Our algorithm is mathematically justified by the observation that the MAP objective can be approximated by a sum of $N$ ``local MAP'' objectives, where $N$ is the number of function evaluations. By leveraging Tweedie's formula, we show that we can perform gradient steps to sequentially optimize these objectives. We validate our approach for various linear inverse problems, such as super-resolution, deblurring, inpainting, and compressed sensing, and demonstrate that we can outperform other methods based on flow matching.

## Envirioment Setup
Clone this repository and create a conda environment:
```
git clone git@github.com:YasminZhang/ICTM.git
conda create -n ictm python=3.9 -y
conda activate ictm
```
Install the following packages:
```{bash}
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 
pip install tensorflow==2.9.0 tensorflow-probability==0.12.2 tensorflow-gan==2.0.0 tensorflow-datasets==4.6.0
pip install jax==0.3.4 jaxlib==0.3.2 
pip install numpy==1.21.6 ninja==1.11.1 matplotlib==3.7.0 ml_collections==0.1.1
pip install tensorflow-io==0.26.0 # https://stackoverflow.com/questions/65623468/unable-to-open-file-libtensorflow-io-so-caused-by-undefined-symbol
```
If the jax or jaxlib installation fails, please use:
```{bash}
pip install jax==VERSION -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install jaxlib==VERSION -f https://storage.googleapis.com/jax-releases/jax_releases.html
```



 

## Checkpoints & Datasets
- flow-checkpoint of CelebAHQ images: please check out the repo [Recitified Flow](https://github.com/gnobitab/RectifiedFlow) or directly use the link [here](https://drive.google.com/file/d/1ryhuJGz75S35GEdWDLiq4XFrsbwPdHnF/view?usp=sharing)
- flow-checkpoint of MRI images: [checkpoint](https://drive.google.com/file/d/16naw5jLuBLe-4IYmC8X-lZKlOFOIxrJh/view?usp=sharing)
- CelebAHQ: we randomly select 100 images of the test set of CelebAHQ dataset
- MRI: we randomly select 200 images of HCP T2w test dataset [link](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release)
 


## ICTM (our method)
 
To run our method, please use the following command:

```
CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/rectified_flow/celeba_hq_pytorch_rf_gaussian_inverse.py --eval_folder <eval_folder> --mode eval_inverse --workdir ./logs/celebahq_ckpt --config.eval.method ours --config.eval.task super_resolution --config.sampling.sample_N 100 --config.eval.eta 1.0e-02  --config.eval.k 1 --config.eval.lamda 1.0e+04 
```
- `task`: super_resolution, inpainting_box, gaussian,  inpainting, cs (compressed sensing)
- `workdir`: where you save the checkpoints
- `config.eval.lamda`: guidance weight
- `config.eval.eta`: step size
- `config.eval.k`: iteration number, default = 1
- `config.sampling.sample_N`: number of sampling steps, default = 100

## Metrics
We mainly use the following metrics to evaluate the generated images:
- PSNR
- SSIM


Please make sure that your `eval_dir` folder structure is as follows:
```
-recon
 --000001.png
 ...
-label
 --000001.png 
 ...
```

To get the PSNR and SSIM scores, (and LPIPS/FID), run the following command:
```{bash}
python get_metric.py --eval_dir=<path/to/evaldir> --enable_fid=<True/False> --gpu=<gpu_id>
```

 

## Baselines
- For Wavelet and TV norms, please check out package [DeepInverse](https://deepinv.github.io/deepinv/index.html)
- For RedDiff and $\Pi$GDM, please check out repo [RED-Diff](https://github.com/NVlabs/RED-diff/tree/master)
- 🔥 For new baselines in flow matching, we refer to this great repo [PnP-flow](https://github.com/annegnx/PnP-Flow) 

## Citation

If you find the code or our results useful, please cite as:

```bibtex
@inproceedings{
zhang2024flow,
title={Flow Priors for Linear Inverse Problems via  Iterative Corrupted Trajectory Matching},
author={Yasi Zhang and Peiyu Yu and Yaxuan Zhu and Yingshan Chang and Feng Gao and Ying Nian Wu and Oscar Leong},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=1H2e7USI09}
}
```



