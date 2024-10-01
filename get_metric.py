import PIL.Image
from cleanfid import fid as fid_fn
import lpips
from pytorch_msssim import ssim
import torch
import PIL
import numpy as np
from tqdm import tqdm
import argparse
import os

def psnr_fn(imageA, imageB):
    """ 
    Compute PSNR between two images 
    default: imageA, imageB are in [-1, 1]
             batch size of imageA and imageB = 1 
    """

    # normalize to [0, 1]
    imageA = (imageA + 1.0) / 2.0
    imageB = (imageB + 1.0) / 2.0
    mse_value = torch.mean((imageA - imageB) ** 2) # actually se instead of mse
    if mse_value == 0:
        return -1  # Means no difference
    psnr_value = 10 * torch.log10(1.0 / mse_value)
    return psnr_value.item()


if __name__ == '__main__':
    
    args = argparse.ArgumentParser()
    args.add_argument('--eval_dir', type=str, default='/home/yasmin/projects/RectifiedFlow/ImageGeneration/logs/celebahq_ckpt/super_resolution/ours_x/')
    args.add_argument('--enable_fid', type=bool, default=False)
    args.add_argument('--lpips', type=str, default='vgg')
    args.add_argument('--gpu', type=int, default=2)
    args = args.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    eval_dir = args.eval_dir
     
    if args.enable_fid:
        # fid score
        fid_score = fid_fn.compute_fid(eval_dir + '/label', eval_dir + '/recon')
        print('FID:', fid_score)     

    psnrs = []
    lpipss = []
    ssims = []
    loss_fn_alex = lpips.LPIPS(net=args.lpips).to('cuda') # best forward scores

    # check out how many images in the eval_dir
    l1, l2 = len(os.listdir(eval_dir + '/label')), len(os.listdir(eval_dir + '/recon'))
    assert l1 == l2, 'The number of images in label and recon are not the same'


    for i in tqdm(range(l1)):
        # read images png from eval_dir using PIL
        label = PIL.Image.open(eval_dir + '/label/' + str(i).zfill(5) + '.png')
        recon = PIL.Image.open(eval_dir + '/recon/' + str(i).zfill(5) + '.png')

        # why does it have 4 channels?
        label = label.convert('RGB')
        recon = recon.convert('RGB')

        # convert to tensor
        label = torch.tensor(np.array(label)).permute(2, 0, 1).unsqueeze(0).float().to('cuda') / 255.0
        recon = torch.tensor(np.array(recon)).permute(2, 0, 1).unsqueeze(0).float().to('cuda') / 255.0

        # compute psnr
        psnr = psnr_fn(label*2-1, recon*2-1)
        psnrs.append(psnr)
        # compute lpips
        lpips_score = loss_fn_alex(label*2-1, recon*2-1).item()
        lpipss.append(lpips_score)
        # compute ssim
        ssim_score = ssim(label, recon, data_range=1.0, size_average=True).item()
        ssims.append(ssim_score)

    print('PSNR:', np.mean(psnrs), 'std:', np.std(psnrs))     
    print('LPIPS:', np.mean(lpipss), 'std:', np.std(lpipss))   
    print('SSIM:', np.mean(ssims), 'std:', np.std(ssims))     

    print(f"{np.mean(lpipss):.3f} +- {np.std(lpipss):.2f}  {np.mean(psnrs):.2f} +- {np.std(psnrs):.2f} {np.mean(ssims):.3f} +- {np.std(ssims):.2f}")



