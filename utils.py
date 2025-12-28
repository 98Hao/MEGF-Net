import time
import torch
import torch.nn as nn
import os
from PIL import Image
import json
import torchvision.transforms.functional as FT
import random
import torch.nn.functional as F
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 常量
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456,
                                   0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224,
                                  0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor(
    [0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor(
    [0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr


def charbonnier_loss(pred, target, eps=1e-6):
    diff=pred-target
    loss=torch.sqrt(diff*diff+eps)
    return loss.mean()

# 定义的SAM损失---两个矢量的角度
class SAMLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred_img, img_tgt):
        """
        Args:
            img_tgt: 真实图像 [B, C, H, W]
            pred_img: 预测图像 [B, C, H, W]
        Returns:
            SAM loss (标量)
        """
        # 确保值在合理范围内
        pred_img = torch.clamp(pred_img, 0, 1)
        img_tgt = torch.clamp(img_tgt, 0, 1)

        # 将通道维度放到最后 [B, H, W, C]
        pred_img = pred_img.permute(0, 2, 3, 1)
        img_tgt = img_tgt.permute(0, 2, 3, 1)

        # 展平空间维度 [B, H*W, C]
        pred_img = pred_img.reshape(pred_img.shape[0], -1, pred_img.shape[-1])
        img_tgt = img_tgt.reshape(img_tgt.shape[0], -1, img_tgt.shape[-1])

        # 计算点积和范数
        dot_product = torch.sum(pred_img * img_tgt, dim=-1)  # [B, H*W]
        norm_pred = torch.norm(pred_img, p=2, dim=-1)  # [B, H*W]
        norm_tgt = torch.norm(img_tgt, p=2, dim=-1)  # [B, H*W]

        # 计算余弦相似度
        cos_theta = dot_product / (norm_pred * norm_tgt + self.eps)
        cos_theta = torch.clamp(cos_theta, -1 + self.eps, 1 - self.eps)

        # 计算SAM (弧度)
        sam_rad = torch.acos(cos_theta)

        # 返回批次平均值
        return torch.mean(sam_rad)


# 定义的刚性损失
class RigidityLoss(nn.Module):
    def __init__(self):
        super(RigidityLoss, self).__init__()

    def forward(self, img1, img2):
        # 计算图像的梯度
        dx1 = img1[:, :, :-1, :] - img1[:, :, 1:, :]  # img1 的 x 方向梯度
        dy1 = img1[:, :, :, :-1] - img1[:, :, :, 1:]  # img1 的 y 方向梯度
        dx2 = img2[:, :, :-1, :] - img2[:, :, 1:, :]  # img2 的 x 方向梯度
        dy2 = img2[:, :, :, :-1] - img2[:, :, :, 1:]  # img2 的 y 方向梯度

        # 确保 dx 和 dy 的形状一致
        dx1 = torch.cat((dx1, torch.zeros_like(dx1[:, :, -1:, :])), dim=2)  # 在最后一行补零
        dy1 = torch.cat((dy1, torch.zeros_like(dy1[:, :, :, -1:])), dim=3)  # 在最后一列补零
        dx2 = torch.cat((dx2, torch.zeros_like(dx2[:, :, -1:, :])), dim=2)  # 在最后一行补零
        dy2 = torch.cat((dy2, torch.zeros_like(dy2[:, :, :, -1:])), dim=3)  # 在最后一列补零

        # 计算雅可比矩阵
        J1 = torch.stack((dx1, dy1), dim=1)  # 第一张图像的雅可比矩阵
        J2 = torch.stack((dx2, dy2), dim=1)  # 第二张图像的雅可比矩阵

        # 计算刚性损失（Frobenius 范数）
        rigidity_loss = torch.mean(torch.norm(J1 - J2, p='fro', dim=(2, 3)))  # 计算 Frobenius 范数并对 batch 维度求平均

        return rigidity_loss


# 定义的旋度和散度损失
class DivergenceCurlLoss(nn.Module):
    def __init__(self):
        super(DivergenceCurlLoss, self).__init__()

    def forward(self, img1, img2):
        # 计算散度
        dx1 = img1[:, :, :-1, :] - img1[:, :, 1:, :]  # img1 的 x 方向梯度
        dy1 = img1[:, :, :, :-1] - img1[:, :, :, 1:]  # img1 的 y 方向梯度
        divergence1 = torch.mean(dx1) + torch.mean(dy1)  # 散度

        dx2 = img2[:, :, :-1, :] - img2[:, :, 1:, :]  # img2 的 x 方向梯度
        dy2 = img2[:, :, :, :-1] - img2[:, :, :, 1:]  # img2 的 y 方向梯度
        divergence2 = torch.mean(dx2) + torch.mean(dy2)  # 散度

        # 计算旋度
        curl1 = (dy1[:, :, :-1, :] - dx1[:, :, :, :-1])  # img1 的旋度
        curl2 = (dy2[:, :, :-1, :] - dx2[:, :, :, :-1])  # img2 的旋度

        # 计算损失
        divergence_loss = torch.mean((divergence1 - divergence2) ** 2)
        curl_loss = torch.mean((curl1 - curl2) ** 2)

        total_loss = divergence_loss + curl_loss
        return total_loss

# 生成数据列表
def create_data_lists(train_folders, test_folders, min_size, output_folder):
    """
    创建训练集和测试集列表文件.
        参数 train_folders: 训练文件夹集合; 各文件夹中的图像将被合并到一个图片列表文件里面
        参数 test_folders: 测试文件夹集合; 每个文件夹将形成一个图片列表文件
        参数 min_size: 图像宽、高的最小容忍值
        参数 output_folder: 最终生成的文件列表,json格式
    """
    print("\n正在创建文件列表... 请耐心等待.\n")
    train_images = list()
    for d in train_folders:
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
    print("训练集中共有 %d 张图像\n" % len(train_images))
    with open(os.path.join(output_folder, 'train_images.json'), 'w') as j:
        json.dump(train_images, j)

    for d in test_folders:
        test_images = list()
        test_name = d.split("/")[-1]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
        print("在测试集 %s 中共有 %d 张图像\n" %
              (test_name, len(test_images)))
        with open(os.path.join(output_folder, test_name + '_test_images.json'),'w') as j:
            json.dump(test_images, j)

    print("生成完毕。训练集和测试集文件列表已保存在 %s 下\n" % output_folder)


def convert_image(img, source, target):
    """
    转换图像格式.

    :参数 img: 输入图像
    :参数 source: 数据源格式, 共有3种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]'
    :参数 target: 数据目标格式, 共5种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]'
                   (4) 'imagenet-norm' (由imagenet数据集的平均值和方差进行标准化)
                   (5) 'y-channel' (亮度通道Y，采用YCbCr颜色空间, 用于计算PSNR 和 SSIM)
    :返回: 转换后的图像
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'
                      }, "无法转换图像源格式 %s!" % source
    assert target in {
        'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm', 'y-channel'
    }, "无法转换图像目标格式t %s!" % target

    # 转换图像数据至 [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)   #把一个取值范围是[0,255]的PIL.Image 转换成形状为[C,H,W]的Tensor，取值范围是[0,1.0]

    elif source == '[0, 1]':
        pass  # 已经在[0, 1]范围内无需处理

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # 从 [0, 1] 转换至目标格式
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # 无需处理

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],
                           rgb_weights) / 255. + 16.

    return img


class ImageTransforms(object):
    """
    图像变换.
    """

    def __init__(self, split, crop_size, scaling_factor, lr_img_type,
                 hr_img_type):
        """
        :参数 split: 'train' 或 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸
        :参数 scaling_factor: 放大比例
        :参数 lr_img_type: 低分辨率图像预处理方式
        :参数 hr_img_type: 高分辨率图像预处理方式
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        """
        对图像进行裁剪和下采样形成低分辨率图像
        :参数 img: 由PIL库读取的图像
        :返回: 特定形式的低分辨率和高分辨率图像
        """

        # 裁剪
        if self.split == 'train':
            # 从原图中随机裁剪一个子块作为高分辨率图像
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # 从图像中尽可能大的裁剪出能被放大比例整除的图像
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # 下采样（双三次差值）
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor),
                                int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)

        # 数据增强，随即旋转和水平反转
        if self.split == 'train':
            # 随即选择一种变换操作
            transform_type = random.choice(['rotate', 'flip', 'none'])
            if transform_type == 'rotate':
                # 随即旋转90,180,270
                angle = random.choice([90,180,270])
                hr_img = hr_img.rotate(angle)
                lr_img = lr_img.rotate(angle)
            elif transform_type == 'flip':
                # 随机水平翻转
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
                lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)

        # 安全性检查
        assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        # 转换图像
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)

        return lr_img, hr_img


'''
生成高斯核和sobel算子---不用了，没有图像增强了
'''
def generate_kernels(kernel_size=15, sigma=2, device="cuda"):
    # Sobel kernel for x direction (horizontal gradient)
    kernel_x = torch.tensor([[[[-1.0, 0.0, 1.0],
                               [-2.0, 0.0, 2.0],
                               [-1.0, 0.0, 1.0]]]], dtype=torch.float32).to(device)

    # Sobel kernel for y direction (vertical gradient)
    kernel_y = torch.tensor([[[[-1.0, -2.0, -1.0],
                               [0.0, 0.0, 0.0],
                               [1.0, 2.0, 1.0]]]], dtype=torch.float32).to(device)

    # Create Gaussian kernel
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32).to(device)
    y = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32).to(device)
    x, y = torch.meshgrid(x, y, indexing="ij")
    gaussian_kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    gaussian_kernel = gaussian_kernel.to(device)

    return gaussian_kernel, kernel_x, kernel_y


'''
生成高斯核和sobel算子---使用这个新的
'''
def generate_kernels_2(device="cuda"):
    # Sobel kernel for x direction (horizontal gradient)
    kernel_x = torch.tensor([[[[-1.0, 0.0, 1.0],
                               [-2.0, 0.0, 2.0],
                               [-1.0, 0.0, 1.0]]]], dtype=torch.float32).to(device)

    # Sobel kernel for y direction (vertical gradient)
    kernel_y = torch.tensor([[[[-1.0, -2.0, -1.0],
                               [0.0, 0.0, 0.0],
                               [1.0, 2.0, 1.0]]]], dtype=torch.float32).to(device)

    return kernel_x, kernel_y



'''
计算散度核旋度
'''
def compute_div_and_curl(input, kernel_x, kernel_y):
    """
    计算输入图像的散度和旋度。

    参数：
    - input: 彩色输入图像，形状为 [batch_size, 3, H, W]
    - kernel_x: Sobel X 内核，形状为 [1, 1, 3, 3]
    - kernel_y: Sobel Y 内核，形状为 [1, 1, 3, 3]

    返回：
    - div: 散度
    - curl: 旋度
    """
    # 转换为灰度图
    input = 0.2989 * input[:, 0:1, :, :] + \
                 0.5870 * input[:, 1:2, :, :] + \
                 0.1140 * input[:, 2:3, :, :]

    # 转换为Y通道
    # input = torch.matmul(input.permute(0, 2, 3, 1), torch.tensor([[0.299, 0.587, 0.114]],device=device).T).permute(0, 3, 1, 2)

    # 使用 Sobel 卷积计算梯度
    dx = F.conv2d(input, weight=kernel_x, padding=1)
    dy = F.conv2d(input, weight=kernel_y, padding=1)

    # 计算散度和旋度
    div = dx + dy
    curl = dy - dx

    # # 释放显存
    # del dx, dy, input
    # torch.cuda.empty_cache()

    return div, curl


# 从lmdb中读取PIL图像
def _read_img_lmdb(env, key):
    """从 LMDB 读取图像，并返回 PIL 格式"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode("ascii"))  # 读取二进制数据
    img = Image.open(io.BytesIO(buf))  # 解析为 PIL 图像
    return img










