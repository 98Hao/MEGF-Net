import torch, os, glob, copy
import torch.nn.functional as F
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision import transforms
from tqdm import tqdm
from torch.backends import cudnn
import psutil
import time
import pyiqa


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True
from model.ours import MYMODEL
mymodel = MYMODEL()
mymodel = mymodel.to(device)
PI_metric = pyiqa.create_metric('pi', device=device).eval()

LR_dir = f'****'
model_dir = f'****'
SR_dir = f'****'

device = torch.device("cuda")

mymodel.load_state_dict(torch.load(model_dir, weights_only=False)['model'])
mymodel = mymodel.to(device)
mymodel.eval()

def calculate_flops_for_image(model, input_tensor):
    flops = 0
    b, c, h, w = input_tensor.shape
    def count_flops(module, input, output):
        nonlocal flops
        if isinstance(module, torch.nn.Conv2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            groups = module.groups
            output_size = output.shape[2] * output.shape[3]

            layer_flops = 2 * (in_channels // groups) * out_channels * kernel_size * output_size
            flops += layer_flops

        elif isinstance(module, torch.nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            batch_size = input[0].shape[0]
            layer_flops = 2 * in_features * out_features * batch_size
            flops += layer_flops

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hook = module.register_forward_hook(count_flops)
            hooks.append(hook)

    with torch.no_grad():
        _ = model(input_tensor)

    for hook in hooks:
        hook.remove()
    return flops

total_inference_time = 0
total_flops = 0
gpu_memory_usages = []
cpu_memory_usages = []

image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.gif', '*.JPG']
test_LR_paths = []
for ext in image_extensions:
    test_LR_paths.extend(glob.glob(os.path.join(LR_dir, ext)))

test_LR_paths = sorted(test_LR_paths)

os.makedirs(SR_dir, exist_ok=True)

with torch.no_grad():
    for i, path in enumerate(tqdm(test_LR_paths)):

        LR = Image.open(path).convert("RGB")
        LR = transforms.ToTensor()(LR).to(device).unsqueeze(0).float()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        initial_gpu_memory = torch.cuda.memory_allocated() / (1024**2)
        initial_cpu_memory = psutil.Process().memory_info().rss / (1024**2)
        current_flops = calculate_flops_for_image(mymodel, LR)
        total_flops += current_flops
        if i == 0:
            with torch.no_grad():
                _ = mymodel(LR)
                torch.cuda.synchronize()
        start_time = time.time()

        SR = mymodel(LR)

        torch.cuda.synchronize()
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000
        total_inference_time += inference_time
        peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)
        current_gpu_memory = torch.cuda.memory_allocated() / (1024**2)
        gpu_memory_usage = current_gpu_memory - initial_gpu_memory
        gpu_memory_usages.append(gpu_memory_usage)
        current_cpu_memory = psutil.Process().memory_info().rss / (1024**2)
        cpu_memory_usage = current_cpu_memory - initial_cpu_memory
        cpu_memory_usages.append(cpu_memory_usage)

        SR = transforms.ToPILImage()(SR[0].clamp(0, 1).cpu())
        SR.save(os.path.join(SR_dir, os.path.basename(path)))


    total_params = sum(p.numel() for p in mymodel.parameters())
    print("模型参数:", str(float(total_params / 1e3)) + 'K')
    avg_flops_per_image = total_flops / len(test_LR_paths)
    print(f"平均每张图像FLOPs: {avg_flops_per_image / 1e9:.2f}G")
    avg_memory = (sum(gpu_memory_usages)+sum(cpu_memory_usages)) / len(test_LR_paths)
    print(f"平均内存使用: {avg_memory:.2f}MB")
    avg_inference_time = total_inference_time / len(test_LR_paths)
    print(f"平均推理时间: {avg_inference_time:.2f}ms")


