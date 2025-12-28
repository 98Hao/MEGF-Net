import torch, os, glob, pyiqa
from argparse import ArgumentParser
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


HR_dir = '***'
SR_dir = '***'
testdataname = '***'

device = torch.device("cuda")

psnr = pyiqa.create_metric("psnr", test_y_channel=False, color_space="rgb", device=device)
ssim = pyiqa.create_metric("ssim", test_y_channel=False, color_space="rgb", device=device)
lpips = pyiqa.create_metric("lpips", device=device)
print('完成lpips')
fid = pyiqa.create_metric("fid", device=device)
print('完成fid')
niqe = pyiqa.create_metric("niqe", device=device)
print('完成niqe')


if testdataname in ['LOLv1', 'LOLv2_Real', 'LOLv2_Synthetic', 'LSRW_HuaWei', 'LSRW_Nikon']:
    test_SR_paths = list(sorted(glob.glob(os.path.join(SR_dir, "*"))))
    test_HR_paths = list(sorted(glob.glob(os.path.join(HR_dir, "*"))))

    metrics = {"psnr": [], "ssim": [], "lpips": []}

    for i, (SR_path, HR_path) in tqdm(enumerate(zip(test_SR_paths, test_HR_paths))):
        SR = Image.open(SR_path).convert("RGB")
        SR = transforms.ToTensor()(SR).to(device).unsqueeze(0)
        HR = Image.open(HR_path).convert("RGB")
        HR = transforms.ToTensor()(HR).to(device).unsqueeze(0)
        metrics["psnr"].append(psnr(SR, HR).item())
        metrics["ssim"].append(ssim(SR, HR).item())
        metrics["lpips"].append(lpips(SR, HR).item())

    for k in metrics.keys():
        metrics[k] = np.mean(metrics[k])

    metrics["fid"] = fid(SR_dir, HR_dir)

    for k, v in metrics.items():
        if k == "niqe":
            print(k, f"{v:.3g}")
        elif k == "fid":
            print(k, f"{v:.5g}")
        else:
            print(k, f"{v:.4g}")
else:
    test_SR_paths = list(sorted(glob.glob(os.path.join(SR_dir, "*"))))
    metrics = {"niqe": []}
    for i, (SR_path) in tqdm(enumerate(zip(test_SR_paths))):
        SR = Image.open(SR_path).convert("RGB")
        SR = transforms.ToTensor()(SR).to(device).unsqueeze(0)
        metrics["niqe"].append(niqe(SR).item())

    for k in metrics.keys():
        metrics[k] = np.mean(metrics[k])

    for k, v in metrics.items():
        if k == "niqe":
            print(k, f"{v:.3g}")
        else:
            print(k, f"{v:.4g}")
