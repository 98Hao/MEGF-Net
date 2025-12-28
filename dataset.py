import torch, random, cv2, os, math, glob
import torch.nn.functional as F
import numpy as np
from bsr.degradations import circular_lowpass_kernel, random_mixed_kernels, random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from bsr.transforms import augment, paired_random_crop
from bsr.utils import FileClient, imfrombytes, img2tensor, DiffJPEG
from bsr.utils.img_process_util import filter2D

class RealESRGANDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, crop_size = 256, split = 'train'):
        super(RealESRGANDataset, self).__init__()
        self.split = split
        self.crop_size = crop_size
        if split == 'train':
            self.file_client = FileClient("disk")
            self.gt_folder = os.path.join(data_folder, 'Train', 'High')
            self.llimg_folder = os.path.join(data_folder, 'Train', 'Low')
            self.gt_paths = sorted(glob.glob(os.path.join(self.gt_folder, "*")))
            self.llimg_paths = sorted(glob.glob(os.path.join(self.llimg_folder, "*")))
            assert len(self.gt_paths) == len(self.llimg_paths), \
            "GT 和 Renders 数量不一致！"
        elif split == 'test1':
            self.file_client = FileClient("disk")
            self.gt_folder = os.path.join(data_folder, 'Test', 'High')
            self.llimg_folder = os.path.join(data_folder, 'Test', 'Low')
            self.gt_paths = sorted(glob.glob(os.path.join(self.gt_folder, "*")))
            self.llimg_paths = sorted(glob.glob(os.path.join(self.llimg_folder, "*")))
            assert len(self.gt_paths) == len(self.llimg_paths), \
            "GT 和 Renders 数量不一致！"
        else:
            self.file_client = FileClient("disk")
            self.llimg_folder = data_folder
            self.llimg_paths = sorted(glob.glob(os.path.join(self.llimg_folder, "*")))

    def __getitem__(self, index):
        if self.split == 'train':
            gt_path = self.gt_paths[index]
            llimg_path = self.llimg_paths[index]
            img_gt = imfrombytes(self.file_client.get(gt_path), float32=True)
            img_renders = imfrombytes(self.file_client.get(llimg_path), float32=True)
            img_gt, img_renders = augment(img_gt, img_renders)
            h, w = img_gt.shape[0:2]
            crop_pad_size = self.crop_size
            if h < crop_pad_size or w < crop_pad_size:
                pad_h = max(0, crop_pad_size - h)
                pad_w = max(0, crop_pad_size - w)
                img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                img_renders = cv2.copyMakeBorder(img_renders, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
                h, w = img_gt.shape[0:2]
                top = random.randint(0, h - crop_pad_size)
                left = random.randint(0, w - crop_pad_size)
                img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
                img_renders = img_renders[top:top + crop_pad_size, left:left + crop_pad_size, ...]

            img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
            img_renders = img2tensor([img_renders], bgr2rgb=True, float32=True)[0]
            return img_renders, img_gt
        elif self.split == 'test1':
            gt_path = self.gt_paths[index]
            llimg_path = self.llimg_paths[index]
            img_name = os.path.basename(self.llimg_paths[index])
            img_gt = imfrombytes(self.file_client.get(gt_path), float32=True)
            img_renders = imfrombytes(self.file_client.get(llimg_path), float32=True)
            img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
            img_renders = img2tensor([img_renders], bgr2rgb=True, float32=True)[0]
            return img_renders, img_gt, img_name
        else:
            llimg_path = self.llimg_paths[index]
            img_name = os.path.basename(self.llimg_paths[index])
            img_renders = imfrombytes(self.file_client.get(llimg_path), float32=True)
            img_renders = img2tensor([img_renders], bgr2rgb=True, float32=True)[0]
            return img_renders, img_name

    def __len__(self):
        return len(self.llimg_paths)
