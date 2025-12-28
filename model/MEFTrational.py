import torch
import torch.nn.functional as F
import math


# ---------- 工具：RGB <-> YCbCr ----------
def rgb_to_ycbcr(x):
    """
    x: [B, 3*N, H, W]
    returns: ycbcr: [B, N, 3, H, W]
    assume input in range [0,1]. If您的MATLAB用[0,255], 先除以255.
    """
    B, Ctot, H, W = x.shape
    assert Ctot % 3 == 0
    N = Ctot // 3
    x = x.view(B, N, 3, H, W)  # [B,N,3,H,W]

    R = x[:, :, 0]
    G = x[:, :, 1]
    Bc = x[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * Bc
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * Bc
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * Bc
    ycbcr = torch.stack([Y, Cb, Cr], dim=2)  # [B,N,3,H,W]
    return ycbcr


def ycbcr_to_rgb(ycbcr):
    """
    ycbcr: [B, 3, H, W] or [B, N, 3, H, W]
    returns: rgb: [B, 3, H, W]
    """
    if len(ycbcr.shape) == 5:
        # [B, N, 3, H, W] -> take first N or average? Let's take first for simplicity
        ycbcr = ycbcr[:, 0]  # [B, 3, H, W]

    Y = ycbcr[:, 0]
    Cb = ycbcr[:, 1]
    Cr = ycbcr[:, 2]

    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb

    rgb = torch.stack([R, G, B], dim=1)  # [B, 3, H, W]
    return torch.clamp(rgb, 0.0, 1.0)


# ---------- contrast, saturation, well_exposedness ----------
def contrast_measure(Y_channel):
    # Y_channel: [B, N, H, W]
    # laplacian kernel
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=Y_channel.device, dtype=Y_channel.dtype).view(
        1, 1, 3, 3)
    B, N, H, W = Y_channel.shape
    # merge batch and N to convolve together
    t = Y_channel.view(B * N, 1, H, W)
    # replicate padding to mimic 'replicate'
    t_pad = F.pad(t, (1, 1, 1, 1), mode='replicate')
    conv = F.conv2d(t_pad, k)
    C = conv.abs().view(B, N, H, W)
    return C


def saturation_measure(ycbcr):
    # ycbcr: [B,N,3,H,W]
    # In MATLAB they used abs(I(:,:,2,i))+abs(I(:,:,3,i))+1
    Cb = ycbcr[:, :, 1]
    Cr = ycbcr[:, :, 2]
    sat = (Cb.abs() + Cr.abs() + 1.0)
    return sat  # [B,N,H,W]


def well_exposedness(Y_channel, sig=0.2):
    C = torch.exp(-0.5 * ((Y_channel - 0.5) ** 2) / (sig ** 2))
    C = torch.clamp(C, min=1e-8)
    return C


# ---------- Gaussian blur kernel (separable approximation) ----------
def gaussian_kernel(device, dtype, ksize=5, sigma=1.0):
    # generate 1D gaussian then outer product
    half = (ksize - 1) // 2
    xs = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g1 = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    g1 = g1 / g1.sum()
    g2 = g1[:, None] * g1[None, :]
    return g2.view(1, 1, ksize, ksize)


def gaussian_blur(x, k):
    # x: [B,1,H,W], k: [1,1,ks,ks]
    pad = (k.shape[-1] - 1) // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    return F.conv2d(x_pad, k)


# ---------- gaussian & laplacian pyramids ----------
def gaussian_pyramid(x, nlev):
    pyr = [x]
    device, dtype = x.device, x.dtype
    k = gaussian_kernel(device, dtype)
    curr = x
    for i in range(1, nlev):
        blurred = gaussian_blur(curr, k)
        curr = F.interpolate(blurred, scale_factor=0.5, mode='bilinear', align_corners=False,
                             recompute_scale_factor=False)
        pyr.append(curr)
    return pyr


def laplacian_pyramid(x, nlev):
    gp = gaussian_pyramid(x, nlev)
    pyr = []
    for i in range(nlev - 1):
        up = F.interpolate(gp[i + 1], size=gp[i].shape[-2:], mode='bilinear', align_corners=False)
        pyr.append(gp[i] - up)
    pyr.append(gp[-1])
    return pyr


def reconstruct_laplacian_pyramid(pyr):
    curr = pyr[-1]
    for lvl in reversed(range(len(pyr) - 1)):
        curr = F.interpolate(curr, size=pyr[lvl].shape[-2:], mode='bilinear', align_corners=False) + pyr[lvl]
    return curr


# ---------- phase consistency filter (approx) ----------
def phase_consistency_filter(images, threshold=0.3):
    """
        images: [B,1,H,W]
        Returns: filtered_image normalized to [0,1], same interface as MATLAB version.
        Implementation: compute local phase via fft, take angle, compute local std in 3x3 window,
        create mask pc>threshold, apply mask in freq domain (approx).
        """
    B, _, H, W = images.shape

    # FFT per image (keep channel dim squeezed for simplicity)
    FF = torch.fft.fft2(images.squeeze(1))  # [B,H,W], complex
    phase = torch.angle(FF)  # [B,H,W]

    # compute local std via 3x3 conv
    phase = phase.unsqueeze(1)  # [B,1,H,W]
    kernel = torch.ones((1, 1, 3, 3), device=images.device, dtype=images.dtype) / 9.0
    pad = 1
    p = F.pad(phase, (pad, pad, pad, pad), mode='replicate')
    mean = F.conv2d(p, kernel)
    mean_sq = F.conv2d(p * p, kernel)
    var = torch.clamp(mean_sq - mean * mean, min=0.0)
    pc = torch.sqrt(var).squeeze(1)  # [B,H,W]

    # create mask
    mask = (pc > threshold).to(images.dtype)  # [B,H,W]

    # apply mask in freq domain
    F_filtered = FF * mask  # [B,H,W]

    # IFFT back to spatial domain
    img_filt = torch.real(torch.fft.ifft2(F_filtered))  # [B,H,W]
    img_filt = img_filt.unsqueeze(1)  # [B,1,H,W]

    # normalize to [0,1]
    mi = img_filt.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
    ma = img_filt.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
    denom = ma - mi
    denom[denom == 0] = 1.0
    img_out = (img_filt - mi) / denom

    return img_out


# ---------- 修复与向量化后的 MEF（稳健版） ----------
def MEF_code_pytorch_fast_fixed(x, threshold=0.3, m=[1, 1, 1], device=None):
    """
    Vectorized, with boundary checks to avoid index errors for small H/W.
    x: [B, 3*N, H, W]
    returns: R: [B,3,H,W]
    """
    if device is None:
        device = x.device
    B, Ctot, H, W = x.shape
    assert Ctot % 3 == 0, "channel num must be multiple of 3"
    N = Ctot // 3
    contrast_parm, sat_parm, wexp_parm = m

    # convert
    ycbcr = rgb_to_ycbcr(x)  # [B,N,3,H,W]
    Y = ycbcr[:, :, 0]  # [B,N,H,W]

    # compute weights WW [B,N,H,W]
    WW = torch.ones_like(Y)
    if contrast_parm > 0:
        WW = WW * (contrast_measure(Y) ** contrast_parm)
    if sat_parm > 0:
        WW = WW * (saturation_measure(ycbcr) ** sat_parm)
    if wexp_parm > 0:
        WW = WW * (well_exposedness(Y) ** wexp_parm)
    WW = WW + 1e-12
    WW = WW / (WW.sum(dim=1, keepdim=True) + 1e-12)
    means2 = (Y.view(B, N, -1).mean(dim=2) ** 2).view(B, N, 1, 1)
    WW = WW * means2
    WW = WW + 1e-12
    WW = WW / (WW.sum(dim=1, keepdim=True) + 1e-12)

    # flatten batch*N
    Y_flat = Y.view(B * N, 1, H, W)  # [B*N,1,H,W]
    WW_flat = WW.view(B * N, 1, H, W)  # [B*N,1,H,W]
    uv_flat = ycbcr[:, :, 1:].reshape(B * N, 2, H, W)  # [B*N,2,H,W]

    # determine nlev but ensure at least 3 to allow nlev-2 indexing
    computed_nlev = int(math.floor(math.log2(max(1, min(H, W)))))
    nlev = max(3, computed_nlev)  # ensure >=3 to safely use index nlev-2

    # --- Y channel blending (vectorized) ---
    pyrY = laplacian_pyramid(Y_flat, nlev)  # list len nlev
    pyrW = gaussian_pyramid(WW_flat, nlev)  # list len nlev

    # accumulate for levels 0..nlev-3
    pyr_acc = [torch.zeros_like(p) for p in pyrY]
    for l in range(0, nlev - 2):
        pyr_acc[l] = pyr_acc[l] + (pyrW[l] * pyrY[l])

    # handle level nlev-2 with phase consistency and high-boost (safe because nlev>=3)
    w_low = phase_consistency_filter(pyrW[nlev - 2], threshold=threshold)  # [B*N,1,h,w]
    I_low = phase_consistency_filter(pyrY[nlev - 2], threshold=threshold)  # [B*N,1,h,w]
    I_highboost = pyrY[nlev - 2] + (pyrY[nlev - 2] - I_low)

    E_flat = well_exposedness(Y_flat)  # [B*N,1,H,W]
    Matris_Ex_resized = F.interpolate(E_flat, size=w_low.shape[-2:], mode='bilinear', align_corners=False)
    mask_ex = (Matris_Ex_resized > threshold).to(x.dtype)

    # combine
    w_low = w_low + 0.01 * I_highboost * mask_ex
    pyr_acc[nlev - 2] = pyr_acc[nlev - 2] + (w_low * pyrY[nlev - 2])

    # Add the lowpass level
    pyr_acc[nlev - 1] = pyr_acc[nlev - 1] + pyrY[nlev - 1]

    # reconstruct RY
    RY_flat = reconstruct_laplacian_pyramid(pyr_acc)  # [B*N,1,H,W]
    RY = RY_flat.view(B, N, H, W)  # [B,N,H,W]

    # --- UV channels blending ---
    nlev_uv = max(1, nlev - 2)
    R_uv_list = []
    for ch in range(2):
        Ci_flat = uv_flat[:, ch:ch + 1, :, :]  # [B*N,1,H,W]
        pyrI = laplacian_pyramid(Ci_flat, nlev_uv)
        pyrW_uv = gaussian_pyramid(WW_flat, nlev_uv)
        pyr_acc_uv = [pyrI[i] * pyrW_uv[i] for i in range(nlev_uv)]
        R_ch = reconstruct_laplacian_pyramid(pyr_acc_uv)  # [B*N,1,H,W]
        R_uv_list.append(R_ch.view(B, N, H, W))  # reshape to [B,N,H,W]

    Rcb = R_uv_list[0]
    Rcr = R_uv_list[1]

    # Sum over N to get final single image
    RY_sum = (RY * WW).sum(dim=1)  # [B,H,W] - weighted sum
    Rcb_sum = (Rcb * WW).sum(dim=1)  # [B,H,W] - weighted sum
    Rcr_sum = (Rcr * WW).sum(dim=1)  # [B,H,W] - weighted sum

    # final stacked result
    R_ycbcr = torch.stack([RY_sum, Rcb_sum, Rcr_sum], dim=1)  # [B,3,H,W]
    R_ycbcr = torch.clamp(R_ycbcr, 0.0, 1.0)

    # Convert back to RGB
    R_rgb = ycbcr_to_rgb(R_ycbcr)
    return R_rgb

