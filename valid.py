import torch
from torch.utils.checkpoint import checkpoint
from torchvision.utils import save_image

from utils import *
import os
from dataset import RealESRGANDataset
import pyiqa
from metrics.image_utils import mse
from metrics.lpipsPyTorch.modules import lpips
import pickle
psnr = pyiqa.create_metric("psnr", test_y_channel=False, color_space="rgb", device=device)
ssim = pyiqa.create_metric("ssim", test_y_channel=False, color_space="rgb", device=device)

def _rundataset(model, testdata_folder):
    psnr_adder_test = Adder()
    ssim_adder_test = Adder()

    test_dataset_test = RealESRGANDataset(data_folder=testdata_folder, split='test1')
    test_loader_test = torch.utils.data.DataLoader(test_dataset_test, batch_size=1)

    total_time = 0
    num_batch = len(test_loader_test)
    with torch.no_grad():
        for idx, (input_img, label_img, name) in enumerate(test_loader_test):

            input_img, label_img = input_img, label_img
            input_img = input_img.to(device).float()
            label_img = label_img.to(device).float()

            start_time = time.time()
            pred_test = model(input_img).float()
            end_time = time.time()

            inference_time_ms = (end_time-start_time)*1000
            total_time += inference_time_ms

            pred_test = torch.clamp(pred_test,0,1)

            ssim1 = ssim(pred_test, label_img).item()
            psnr_test = psnr(pred_test, label_img).item()

            psnr_adder_test(psnr_test)
            ssim_adder_test(ssim1)

    avg_total_time = total_time/num_batch
    return psnr_adder_test.average(), ssim_adder_test.average(), avg_total_time



def _valid(model, testdata_folder):
    model.eval()
    Set5psnr, Set5ssim, Set5time =_rundataset(model, testdata_folder)
    return Set5psnr, Set5ssim

