import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

''' Misc
'''
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

''' Checkpoint utils
'''
def load_checkpoint(model, optimizer, ckpt_path, no_reload_optimizer):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    start = ckpt['global_step']
    model.load_state_dict(ckpt['model_state_dict'])
    if not no_reload_optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    # scaler.load_state_dict(ckpt['scaler_state_dict'])
    return model, optimizer, start

def load_model(model_class, ckpt_path, strict=True):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)  
    model = model_class(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'], strict=strict)
    return model

''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()

''' Ray and batch
'''
@torch.no_grad()
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d

@torch.no_grad()
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d

@torch.no_grad()
def get_rays_minimalized(H, W, K, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device)) 
    i = i.t().float()
    j = j.t().float()
    i, j = i+0.5, j+0.5
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d

@torch.no_grad()
def rayPlaneInter(n, p0, ray_o, ray_d):
    s1 = torch.sum(p0 * n, dim=1)
    s2 = torch.sum(ray_o * n, dim=1)
    s3 = torch.sum(ray_d * n, dim=1)
    dist = (s1 - s2) / s3
    dist_group = torch.broadcast_to(dist,(3,dist.shape[0])).T
    inter_point = ray_o + dist_group * ray_d
    return inter_point

@torch.no_grad()
def get_rays_of_a_view_twoplane(H, W, K, c2w):
# def get_rays_of_a_view_twoplane(H, W, K, c2w, ndc, focal_depth):
    # rays_o, rays_d = get_rays_minimalized(H, W, K, c2w)
    # # rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    # rays_o = rays_o.flatten(0,1)
    # rays_d = rays_d.flatten(0,1)
    # plane_normal = torch.tensor([0.0, 0.0, 1.0]).expand(H*W, -1)
    # p_uv = torch.tensor([0.0, 0.0, 0.0]).expand(H*W, -1)
    # p_st = torch.tensor([0.0, 0.0, -2.5]).expand(H*W, -1)
    # inter_uv = rayPlaneInter(plane_normal,p_uv,rays_o,rays_d)
    # inter_st = rayPlaneInter(plane_normal,p_st,rays_o,rays_d)
    # ray = torch.cat((inter_uv[:,:2], inter_st[:,:2]), dim=1)
    # viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    # return ray, viewdirs
    # xy = c2w[:2,3].expand(H*W, -1)
    # i, j = torch.meshgrid(
    #     torch.linspace(-1, 1, W, device=c2w.device),
    #     torch.linspace(-1, 1, H, device=c2w.device),
    #     indexing='xy')
    # i = i.flatten().reshape(-1,1)
    # j = j.flatten().reshape(-1,1)
    # dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    # rays_d = rays_d.flatten(0,1)
    # viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    # ray = torch.cat((xy, i, j), dim=1)
    
    uv_scale = 1.0
    st_scale = 0.25
    u = torch.linspace(-1, 1, W, dtype=torch.float32)
    v = torch.linspace(1, -1, H, dtype=torch.float32)
    # v = torch.linspace(1, -1, H, dtype=torch.float32) / aspect
    
    vu = list(torch.meshgrid([v, u]))
    u = vu[1] * uv_scale
    v = vu[0] * uv_scale
    s = torch.ones_like(vu[1]) * c2w[0,3] * st_scale
    t = torch.ones_like(vu[0]) * c2w[1,3] * st_scale
    ray = torch.stack([s, t, u, v], dim=-1).view(-1, 4)
    
    return ray

@torch.no_grad()
def get_rays_of_a_view_plucker(H, W, K, c2w):
    rays_o, rays_d = get_rays_minimalized(H, W, K, c2w)
    rays_o = rays_o.flatten(0,1)
    rays_d = rays_d.flatten(0,1)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    ray = torch.cat((rays_o, viewdirs), dim=1)
    return ray

@torch.no_grad()
def get_ray_min_max(ray_type, rgb_all, train_poses, HW, Ks):
    assert len(rgb_all) == len(train_poses) and len(rgb_all) == len(Ks) and len(rgb_all) == len(HW)
    DEVICE = rgb_all[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_all)
    top = 0
    n = HW[0,0] * HW[0,1]
    if ray_type == 'twoplane':
        ray_all = torch.zeros([N, 4], device=DEVICE)
        for c2w, img, (H, W), K in zip(train_poses, rgb_all, HW, Ks):
            assert img.shape[:2] == (H, W)
            ray = get_rays_of_a_view_twoplane(
                    H=H, W=W, K=K, c2w=c2w)  
            ray_all[top:top+n].copy_(ray)
            top += n
        assert top == N
    elif ray_type == 'plucker':
        ray_all = torch.zeros([N, 4], device=DEVICE)
        for c2w, img, (H, W), K in zip(train_poses, rgb_all, HW, Ks):
            assert img.shape[:2] == (H, W)
            ray = get_rays_of_a_view_plucker(
                    H=H, W=W, K=K, c2w=c2w)
            ray_all[top:top+n].copy_(ray)
            top += n
        assert top == N
    else:
        raise NotImplementedError
    
    ray_min = ray_all.min(0).values
    ray_max = ray_all.max(0).values
    print(f'ray_all min: {ray_min}, ray_all max: {ray_max}')
    unique = [ray_all[:, n].unique().numel() for n in range(ray_all.shape[1])]
    print(f'ray_all unique: {unique}')
    return ray_min, ray_max

@torch.no_grad()
def get_training_rays(ray_type, rgb_original, train_poses, HW, Ks):
    assert len(rgb_original) == len(train_poses) and len(rgb_original) == len(Ks) and len(rgb_original) == len(HW)
    DEVICE = rgb_original[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_original)
    rgb_train = torch.zeros([N, 3], device=DEVICE)
    top = 0
    n = HW[0,0] * HW[0,1]
    if ray_type == 'twoplane':
        ray_train = torch.zeros([N, 4], device=DEVICE)
        # viewdirs_train = torch.zeros([N, 3], device=DEVICE)
        for c2w, img, (H, W), K in zip(train_poses, rgb_original, HW, Ks):
            assert img.shape[:2] == (H, W)
            ray = get_rays_of_a_view_twoplane(
                    H=H, W=W, K=K, c2w=c2w) 
            rgb_train[top:top+n].copy_(img.flatten(0,1))
            ray_train[top:top+n].copy_(ray)
            # viewdirs_train[top:top+n].copy_(viewdirs)
            top += n
        assert top == N
    elif ray_type == 'plucker':
        ray_train = torch.zeros([N, 6], device=DEVICE)
        for c2w, img, (H, W), K in zip(train_poses, rgb_original, HW, Ks):
            assert img.shape[:2] == (H, W)
            ray = get_rays_of_a_view_plucker(
                    H=H, W=W, K=K, c2w=c2w)
            rgb_train[top:top+n].copy_(img.flatten(0,1))
            ray_train[top:top+n].copy_(ray)
            top += n
        assert top == N
    else:
        raise NotImplementedError
    
    ray_min = ray_train.min(0).values
    ray_max = ray_train.max(0).values
    print(f'ray_train min: {ray_min}, ray_train max: {ray_max}')
    unique = [ray_train[:, n].unique().numel() for n in range(ray_train.shape[1])]
    print(f'ray_train unique: {unique}')
    # viewdirs_min = viewdirs_train.min(0).values
    # viewdirs_max = viewdirs_train.max(0).values
    # print(f'viewdirs_train min: {viewdirs_min}, viewdirs_train max: {viewdirs_max}')
    # unique = [viewdirs_train[:, n].unique().numel() for n in range(viewdirs_train.shape[1])]
    # print(f'viewdirs_train unique: {unique}')
    return rgb_train, ray_train

def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS