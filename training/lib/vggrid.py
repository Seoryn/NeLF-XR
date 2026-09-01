import os
import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F
import time

from torch.autograd import gradcheck

# import grid_interpole
try:
    import grid_interpole as _gi
    _HAS_GI = True
except Exception as e:
    _gi = None
    _HAS_GI = False
    print(f"[WARN] grid_interpole failed to load: {e} — using PyTorch fallback kernels.")

# from torch.utils.cpp_extension import load
# parent_dir = os.path.dirname(os.path.abspath(__file__))

# grid_interpole_cuda = load(
#         name='grid_interpole_cuda',
#         sources=[
#             os.path.join(parent_dir, path)
#             for path in ['cuda/4d_interpole.cpp', 'cuda/4d_interpole.cu']],
#         verbose=True)


# ''' grid_interpole
# ''' 

# class gridinterpole_1d(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, grid: torch.Tensor, rays: torch.Tensor):
#         """
#         grid: [C, U]
#         rays: [N, 1]
#         """
#         N = rays.shape[0]
#         C = grid.shape[0]
#         out = torch.empty((N, C), dtype=grid.dtype, device=grid.device)

#         # forward 호출
#         grid_interpole_cuda.forward_1d(grid, rays, out)
        
#         # backward에 필요한 텐서 저장
#         ctx.save_for_backward(rays)
#         ctx.grid_shape = grid.shape  # (C, U)
#         return out
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         grad_output: [N, C]
#         returns: dgrid, drays(없으면 None)
#         """
#         (rays,) = ctx.saved_tensors 
#         (C, U) = ctx.grid_shape

#         # grid에 대한 grad
#         grad_grid = torch.zeros((C, U), dtype=grad_output.dtype, device=grad_output.device)

#         # 실제 backward cuda 호출
#         grad_output = grad_output.contiguous()
#         grid_interpole_cuda.backward_1d(grad_output, rays, grad_grid)
#         # breakpoint()
#         # print(grad_grid)
#         # 이 예시에선 rays에 대한 미분(좌표 미분)은 구현 안 했으므로 None
#         return grad_grid, None
    

# class gridinterpole_4d(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, grid: torch.Tensor, rays: torch.Tensor):
#         """
#         grid: [C, U, V, S, T]
#         rays: [N, 4], in [0,1]
#         returns: [N, C]
#         """
#         N = rays.shape[0]
#         C = grid.shape[0]
#         out = torch.empty((N, C), dtype=grid.dtype, device=grid.device)

#         # forward 호출
#         grid_interpole_cuda.forward_4d(grid, rays, out)

#         # backward에 필요한 텐서 저장
#         ctx.save_for_backward(rays)
#         ctx.grid_shape = grid.shape  # (C, U, V, S, T)
#         return out
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         grad_output: [N, C]
#         returns: dgrid, drays(없으면 None)
#         """
#         (rays,) = ctx.saved_tensors
#         (C, U, V, S, T) = ctx.grid_shape

#         # grid에 대한 grad
#         grad_grid = torch.zeros((C, U, V, S, T), dtype=grad_output.dtype, device=grad_output.device)

#         # 실제 backward cuda 호출
#         # breakpoint()
#         grad_output = grad_output.contiguous()
#         grid_interpole_cuda.backward_4d(grad_output, rays, grad_grid)
#         # breakpoint()
#         # print(grad_grid)
#         # 이 예시에선 rays에 대한 미분(좌표 미분)은 구현 안 했으므로 None
#         return grad_grid, None
    
# def grid_interpolate_1d(grid: torch.Tensor, rays: torch.Tensor):
#     """
#     편의 함수:
#     grid: [C, U]
#     rays: [N, 1]  (normalized coords)
#     -> returns [N, C]
#     """
#     return gridinterpole_1d.apply(grid, rays)
    
# def grid_interpolate_4d(grid: torch.Tensor, rays: torch.Tensor):
#     """
#     편의 함수:
#     grid: [C, U, V, S, T]
#     rays: [N, 4]  (normalized coords)
#     -> returns [N, C]
#     """
#     return gridinterpole_4d.apply(grid, rays)

def create_grid(type, **kwargs):
    if type == 'LF2DGrid':
        # # return LF2DGrid(**kwargs)
        return LF2DGrid_ONNXSafe(**kwargs)
    elif type == 'LF3DGrid':
        # return LF3DGrid(**kwargs)
        return LF3DGrid_ONNXSafe(**kwargs)
    elif type == 'LF4DGrid':
        if 'interp_mask' in kwargs:
            return LF4DGrid_ONNXSafe_interpolation(**kwargs)
        else:
            # return LF4DGrid(**kwargs)
            return LF4DGrid_ONNXSafe(**kwargs)
    elif type == 'LF5DGrid':
        return LF5DGrid(**kwargs)
    elif type == 'LF1DGrid':
        return LF1DGrid(**kwargs)
    else:
        raise NotImplementedError
    
''' LF 1D grid
'''
class LF1DGrid(nn.Module):
    def __init__(self, channels, ray_min, ray_max, grid_size, **kwargs):
        super(LF1DGrid, self).__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.register_buffer('ray_min', torch.Tensor(ray_min))
        self.register_buffer('ray_max', torch.Tensor(ray_max))
        self.grid = nn.Parameter(torch.zeros([channels, grid_size]))
        
    def forward(self, ray):
        ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min)) * 2 - 1
        # out = self.interpolate_1d(ind_norm)
        out = grid_interpolate_1d(self.grid, ind_norm)
        return out
    
    def interpolate_1d(self, grid_indices):
        bottom_indices = grid_indices.floor().long()
        top_indices = bottom_indices + 1
        weights = grid_indices - bottom_indices.float()
        omw = 1 - weights
        u0 = self.grid[:, bottom_indices]
        u1 = self.grid[:, top_indices]
        interpolated = (u0 * omw + u1 * weights).squeeze().T
        return interpolated
    
    def extra_repr(self):
        return f'channels={self.channels}, grid_size={self.grid_size.tolist()}'

''' LF 2D grid
'''
class LF2DGrid(nn.Module):
    def __init__(self, channels, ray_min, ray_max, grid_size, **kwargs):
        super(LF2DGrid, self).__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.register_buffer('ray_min', torch.Tensor(ray_min))
        self.register_buffer('ray_max', torch.Tensor(ray_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *grid_size]))
        # self.grid = nn.Parameter(torch.zeros([channels, *grid_size]))
        
    def forward(self, ray):
        # ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min))
        # grid_indices = (ind_norm * (self.grid_size - 1))
        # out = self.interpolate_2d(grid_indices)
        shape = ray.shape[:-1]
        ray = ray.reshape(1,1,-1,2)
        ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        # if self.channels == 1:
        #     out = out.squeeze(-1)
        return out
    
    def interpolate_2d(self, grid_indices):
        
        bottom_indices = grid_indices.floor().long()
        top_indices = bottom_indices + 1
        weights = grid_indices - bottom_indices.float()
        omw = torch.ones_like(weights) - weights
        
        def gather_values(mask, indices):
            result = torch.empty((self.channels, grid_indices.shape[0]), dtype=self.grid.dtype, device=self.grid.device)
            valid_indices = indices[mask]
            if valid_indices.size(0) > 0:
                result[:, mask] = self.grid[:, valid_indices[:, 0], valid_indices[:, 1]]
            return result
        
        # Precompute all possible index combinations
        all_combinations = [
            bottom_indices,
            torch.stack([top_indices[:, 0], bottom_indices[:, 1]], dim=1),
            torch.stack([bottom_indices[:, 0], top_indices[:, 1]], dim=1),
            top_indices
        ]
        
        # Gather all needed grid values
        u_values = []
        for indices in all_combinations:
            valid_mask = ((indices >= 0) & (indices < self.grid_size)).all(dim=-1)
            u_values.append(gather_values(valid_mask, indices))
            
        # Compute interpolations
        out = (
            ((u_values[0] * omw[:, 0] + u_values[1] * weights[:, 0]) * omw[:, 1] +
            (u_values[2] * omw[:, 0] + u_values[3] * weights[:, 0]) * weights[:, 1]
        ).T)
        return out
    
    def scale_volume_grid(self, new_grid_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_grid_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_grid_size), mode='bilinear', align_corners=True))
            
    def compute_plane_tv(self):
        # breakpoint()
        batch_size, c, h, w = self.grid.shape
        count_h = batch_size * c * (h - 1) * w
        count_w = batch_size * c * h * (w - 1)
        h_tv = torch.square(self.grid[..., 1:, :] - self.grid[..., :h-1, :]).sum()
        w_tv = torch.square(self.grid[..., :, 1:] - self.grid[..., :, :w-1]).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) 
    
    # def compute_plane_smoothness(self.grid):
    #     batch_size, c, h, w = self.grid.shape
    #     # Convolve with a second derivative filter, in the time dimension which is dimension 2
    #     first_difference = self.grid[..., 1:, :] - self.grid[..., :h-1, :]  # [batch, c, h-1, w]
    #     second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
    #     # Take the L2 norm of the result
    #     return torch.square(second_difference).mean()
    
    def extra_repr(self):
        return f'channels={self.channels}, grid_size={self.grid_size.tolist()}'
    
class LF2DGrid_ONNXSafe(nn.Module):
    def __init__(self, channels, ray_min, ray_max, grid_size, interp_mask=(1,1), **kwargs):
        super().__init__()
        self.channels = channels
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.long))  # [U, V]
        self.register_buffer('ray_min', torch.tensor(ray_min, dtype=torch.float32))   # [2]
        self.register_buffer('ray_max', torch.tensor(ray_max, dtype=torch.float32))   # [2]

        C, U, V = int(channels), int(grid_size[0]), int(grid_size[1])
        self.grid = nn.Parameter(torch.zeros([C, U, V], dtype=torch.float32))   # [C, U, V]

        # trace-safe mask
        im_tuple = tuple(bool(int(x)) for x in interp_mask)
        assert len(im_tuple) == 2, "interp_mask must be (U,V)."
        self.im0, self.im1 = im_tuple
        self.register_buffer('interp_mask_tensor', torch.tensor(im_tuple, dtype=torch.bool))

    # [1, C, U, V] → [C, U, V]
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        k = prefix + 'grid'
        if k in state_dict:
            v = state_dict[k]
            if isinstance(v, torch.Tensor) and v.dim() == 4 and v.shape[0] == 1:
                state_dict[k] = v.squeeze(0).contiguous()     # [C, U, V]
            if isinstance(state_dict[k], torch.Tensor) and state_dict[k].dtype != self.grid.dtype:
                state_dict[k] = state_dict[k].to(self.grid.dtype)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, ray):  # ray: [N, 2]
        # 좌표 정규화
        ray = ray.to(self.ray_min.dtype)
        rn = (ray - self.ray_min) / (self.ray_max - self.ray_min)  # [N, 2]
        gs_f = self.grid_size.to(ray.device).float()
        gi = rn * (gs_f - 1.0)  # [N, 2]

        # Clamp 경계 처리
        max_idx = (self.grid_size - 2).to(ray.device).float()
        gi = torch.minimum(torch.maximum(gi, torch.zeros_like(gi)),
                           max_idx.view(1, 2).expand_as(gi))

        # 정수 / 소수 가중치 분리
        i0 = gi.floor().long()    # [N, 2]
        i1 = i0 + 1
        w  = gi - i0.float()      # [N, 2]
        ow = 1.0 - w

        # Nearest indices for axes without interpolation
        grid_idx_near = rn * (gs_f - 1.0)
        max_idx_near = (self.grid_size - 1).to(ray.device).float()
        grid_idx_near = torch.minimum(torch.maximum(grid_idx_near, torch.zeros_like(grid_idx_near)),
                                      max_idx_near.view(1, 2).expand_as(grid_idx_near))
        nn_idx = grid_idx_near.floor().to(torch.long)

        # 그리드 평탄화
        C, U, V = self.grid.shape
        flat = self.grid.view(C, -1).permute(1, 0).contiguous()  # [U*V, C]

        # 선형 인덱스
        def lin_idx(ix, iy): 
            return ix * V + iy  # [N]

        # 가중치 합 계산
        out = torch.zeros(ray.shape[0], self.channels, device=ray.device, dtype=self.grid.dtype)

        m0, m1 = self.im0, self.im1  # Python bools (trace-safe)
        for bu in (0, 1):
            for bv in (0, 1):
                active = 1.0
                if (not m0) and bu == 1: active = 0.0
                if (not m1) and bv == 1: active = 0.0
                if active == 0.0:
                    continue

                wu = ow[:, 0] if bu == 0 else w[:, 0]
                wv = ow[:, 1] if bv == 0 else w[:, 1]
                wt = wu * wv  # [N]

                iu = i0[:, 0] if bu == 0 else i1[:, 0]
                iv = i0[:, 1] if bv == 0 else i1[:, 1]

                lin = lin_idx(iu, iv)
                lin = torch.clamp(lin, 0, flat.shape[0] - 1)
                lin_expand = lin.unsqueeze(1).expand(-1, self.channels)
                gathered = torch.gather(flat, dim=0, index=lin_expand)  # [N, C]
                out = out + gathered * wt.unsqueeze(1)

        return out  # [N, C]

    def extra_repr(self):
        return f'channels={self.channels}, grid_size={self.grid_size.tolist()}'

''' LF 3D grid
'''
class LF3DGrid(nn.Module):
    def __init__(self, channels, ray_min, ray_max, grid_size, **kwargs):
        super(LF3DGrid, self).__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.register_buffer('ray_min', torch.Tensor(ray_min))
        self.register_buffer('ray_max', torch.Tensor(ray_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *grid_size]))
        # self.grid = nn.Parameter(torch.zeros([channels, *grid_size]))
        # self.affine_mlp = nn.Sequential(
        #     nn.Linear(3, 128), nn.ReLU(inplace=True),
        #     *[
        #         nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        #     ],
        #     nn.Linear(128, 3*3+3),
        # )
        
    def forward(self, ray):
        # pytorch 기본 함수 구현
        # ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min))
        # grid_indices = (ind_norm * (self.grid_size - 1))
        # out = self.interpolate_3d(grid_indices)
        
        shape = ray.shape[:-1]
        ray = ray.reshape(1,1,1,-1,3) # [1, 1, 1, N, 3]
        ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        # if self.channels == 1:
        #     out = out.squeeze(-1)

        # CUDA 구현
        # ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min)) * 2 - 1
        # out = grid_interpole.grid_interpolate_3d(self.grid[0], ind_norm)
        return out
    
    def interpolate_3d(self, grid_indices):
        bottom_indices = grid_indices.floor().long()  # [N, 3]
        top_indices = bottom_indices + 1  # [N, 3]
        weights = grid_indices - bottom_indices.float()  # [N, 3]
        omw = torch.ones_like(weights) - weights  # [N, 3]
        
        def gather_values(mask, indices):
            result = torch.empty((self.channels, grid_indices.shape[0]), dtype=self.grid.dtype, device=self.grid.device)
            valid_indices = indices[mask]
            if valid_indices.size(0) > 0:
                result[:, mask] = self.grid[:, valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]
            return result
        
        # Precompute all possible index combinations
        all_combinations = [
            bottom_indices,
            torch.stack([top_indices[:, 0], bottom_indices[:, 1], bottom_indices[:, 2]], dim=1),
            torch.stack([bottom_indices[:, 0], top_indices[:, 1], bottom_indices[:, 2]], dim=1),
            torch.stack([top_indices[:, 0], top_indices[:, 1], bottom_indices[:, 2]], dim=1),
            torch.stack([bottom_indices[:, 0], bottom_indices[:, 1], top_indices[:, 2]], dim=1),
            torch.stack([top_indices[:, 0], bottom_indices[:, 1], top_indices[:, 2]], dim=1),
            torch.stack([bottom_indices[:, 0], top_indices[:, 1], top_indices[:, 2]], dim=1),
            torch.stack([top_indices[:, 0], top_indices[:, 1], top_indices[:, 2]], dim=1),
            top_indices
        ]
        
        # Gather all needed grid values
        u_values = []
        for indices in all_combinations:
            valid_mask = ((indices >= 0) & (indices < self.grid_size)).all(dim=-1)
            u_values.append(gather_values(valid_mask, indices))
            
        # Compute interpolations
        out = (
            (((u_values[0] * omw[:, 0] + u_values[1] * weights[:, 0]) * omw[:, 1] +
            (u_values[2] * omw[:, 0] + u_values[3] * weights[:, 0]) * weights[:, 1]) * omw[:, 2] +
            ((u_values[4] * omw[:, 0] + u_values[5] * weights[:, 0]) * omw[:, 1] +
            (u_values[6] * omw[:, 0] + u_values[7] * weights[:, 0]) * weights[:, 1]) * weights[:, 2]
        ).T)
        return out
        
    def scale_volume_grid(self, new_grid_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_grid_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_grid_size), mode='trilinear', align_corners=True))
            
    def extra_repr(self):
        return f'channels={self.channels}, grid_size={self.grid_size.tolist()}'
    
class LF3DGrid_ONNXSafe(nn.Module):
    def __init__(self, channels, ray_min, ray_max, grid_size, interp_mask=(1,1,1), **kwargs):
        super().__init__()
        self.channels = channels
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.long))  # [U, V, S]
        self.register_buffer('ray_min', torch.tensor(ray_min, dtype=torch.float32))   # [3]
        self.register_buffer('ray_max', torch.tensor(ray_max, dtype=torch.float32))   # [3]
        C, U, V, S = int(channels), int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
        self.grid = nn.Parameter(torch.zeros([C, U, V, S], dtype=torch.float32))      # [C, U, V, S]

        # trace-safe mask
        im_tuple = tuple(bool(int(x)) for x in interp_mask)
        assert len(im_tuple) == 3, "interp_mask must be (U,V,S)."
        self.im0, self.im1, self.im2 = im_tuple
        self.register_buffer('interp_mask_tensor', torch.tensor(im_tuple, dtype=torch.bool))

    # [1, C, U, V, S] → [C, U, V, S]
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        k = prefix + 'grid'
        if k in state_dict:
            v = state_dict[k]
            if isinstance(v, torch.Tensor) and v.dim() == 5 and v.shape[0] == 1:
                state_dict[k] = v.squeeze(0).contiguous()
            if isinstance(state_dict[k], torch.Tensor) and state_dict[k].dtype != self.grid.dtype:
                state_dict[k] = state_dict[k].to(self.grid.dtype)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, ray):  # ray: [N, 3]
        # 좌표 정규화
        ray = ray.to(self.ray_min.dtype)
        rn = (ray - self.ray_min) / (self.ray_max - self.ray_min)  # [N, 3]
        gs_f = self.grid_size.to(ray.device).float()
        gi = rn * (gs_f - 1.0)  # [N, 3]

        # Clamp 경계 처리
        max_idx = (self.grid_size - 2).to(ray.device).float()
        gi = torch.minimum(torch.maximum(gi, torch.zeros_like(gi)),
                           max_idx.view(1, 3).expand_as(gi))

        # 정수 / 소수 가중치 분리
        i0 = gi.floor().long()
        i1 = i0 + 1
        w  = gi - i0.float()
        ow = 1.0 - w

        # Nearest indices for axes without interpolation
        grid_idx_near = rn * (gs_f - 1.0)
        max_idx_near = (self.grid_size - 1).to(ray.device).float()
        grid_idx_near = torch.minimum(torch.maximum(grid_idx_near, torch.zeros_like(grid_idx_near)),
                                      max_idx_near.view(1, 3).expand_as(grid_idx_near))
        nn_idx = grid_idx_near.floor().to(torch.long)

        # 그리드 평탄화
        C, U, V, S = self.grid.shape
        flat = self.grid.view(C, -1).permute(1, 0).contiguous()  # [U*V*S, C]

        # 선형 인덱스
        def lin_idx(ix, iy, iz):
            return ix*(V*S) + iy*S + iz  # [N]

        # 가중치 합 계산
        out = torch.zeros(ray.shape[0], self.channels, device=ray.device, dtype=self.grid.dtype)
        
        m0, m1, m2 = self.im0, self.im1, self.im2  # Python bools (trace-safe)

        for bu in (0, 1):
            for bv in (0, 1):
                for bs in (0, 1):
                    active = 1.0
                    if (not m0) and bu == 1: active = 0.0
                    if (not m1) and bv == 1: active = 0.0
                    if (not m2) and bs == 1: active = 0.0
                    if active == 0.0:
                        continue

                    wu = ow[:, 0] if bu == 0 else w[:, 0]
                    wv = ow[:, 1] if bv == 0 else w[:, 1]
                    ws = ow[:, 2] if bs == 0 else w[:, 2]
                    wt = wu * wv * ws  # [N]

                    iu = i0[:, 0] if bu == 0 else i1[:, 0]
                    iv = i0[:, 1] if bv == 0 else i1[:, 1]
                    iz = i0[:, 2] if bs == 0 else i1[:, 2]

                    lin = lin_idx(iu, iv, iz)
                    lin = torch.clamp(lin, 0, flat.shape[0] - 1)
                    lin_expand = lin.unsqueeze(1).expand(-1, self.channels)
                    gathered = torch.gather(flat, dim=0, index=lin_expand)  # [N, C]
                    out = out + gathered * wt.unsqueeze(1)

        return out  # [N, C]

    def extra_repr(self):
        return f'channels={self.channels}, grid_size={self.grid_size.tolist()}'

''' LF 4D grid
'''
class LF4DGrid(nn.Module):
    def __init__(self, channels, ray_min, ray_max, grid_size, **kwargs):
        super(LF4DGrid, self).__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.register_buffer('ray_min', torch.Tensor(ray_min))
        self.register_buffer('ray_max', torch.Tensor(ray_max))
        self.grid = nn.Parameter(torch.zeros([channels, *grid_size]).requires_grad_(True))
        # self.grid2 = nn.Parameter(torch.zeros([channels, *grid_size]))
        
    def forward(self, ray):
        ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min)) * 2 - 1
        # grid_indices = (ind_norm * (self.grid_size - 1))
        # out = self.interpolate_4d(grid_indices)
        out = grid_interpole.grid_interpolate_4d(self.grid, ind_norm)
        # print(torch.allclose(out, out2, atol=1e-4))
        # F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        # test = gradcheck(grid_interpolate_4d, (self.grid, ray))
        # print(test)
        return out

    def interpolate_4d(self, grid_indices):
        bottom_indices = grid_indices.floor().long()  # [N, 4]
        top_indices = bottom_indices + 1  # [N, 4]
        weights = grid_indices - bottom_indices # [N, 4]
        omw = torch.ones_like(weights) - weights
        
        def gather_values(mask, indices):
            result = torch.empty((self.channels, grid_indices.shape[0]), dtype=self.grid.dtype, device=self.grid.device)
            valid_indices = indices[mask]
            if valid_indices.size(0) > 0:
                result[:, mask] = self.grid[:, valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2], valid_indices[:, 3]]
            return result
        
        # Precompute all possible index combinations
        all_combinations = [
            bottom_indices,
            torch.stack([top_indices[:, 0], bottom_indices[:, 1], bottom_indices[:, 2], bottom_indices[:, 3]], dim=1),
            torch.stack([bottom_indices[:, 0], top_indices[:, 1], bottom_indices[:, 2], bottom_indices[:, 3]], dim=1),
            torch.stack([top_indices[:, 0], top_indices[:, 1], bottom_indices[:, 2], bottom_indices[:, 3]], dim=1),
            torch.stack([bottom_indices[:, 0], bottom_indices[:, 1], top_indices[:, 2], bottom_indices[:, 3]], dim=1),
            torch.stack([top_indices[:, 0], bottom_indices[:, 1], top_indices[:, 2], bottom_indices[:, 3]], dim=1),
            torch.stack([bottom_indices[:, 0], top_indices[:, 1], top_indices[:, 2], bottom_indices[:, 3]], dim=1),
            torch.stack([top_indices[:, 0], top_indices[:, 1], top_indices[:, 2], bottom_indices[:, 3]], dim=1),
            torch.stack([bottom_indices[:, 0], bottom_indices[:, 1], bottom_indices[:, 2], top_indices[:, 3]], dim=1),
            torch.stack([top_indices[:, 0], bottom_indices[:, 1], bottom_indices[:, 2], top_indices[:, 3]], dim=1),
            torch.stack([bottom_indices[:, 0], top_indices[:, 1], bottom_indices[:, 2], top_indices[:, 3]], dim=1),
            torch.stack([top_indices[:, 0], top_indices[:, 1], bottom_indices[:, 2], top_indices[:, 3]], dim=1),
            torch.stack([bottom_indices[:, 0], bottom_indices[:, 1], top_indices[:, 2], top_indices[:, 3]], dim=1),
            torch.stack([top_indices[:, 0], bottom_indices[:, 1], top_indices[:, 2], top_indices[:, 3]], dim=1),
            torch.stack([bottom_indices[:, 0], top_indices[:, 1], top_indices[:, 2], top_indices[:, 3]], dim=1),
            top_indices
        ]
        breakpoint()
        # Gather all needed grid values
        u_values = []
        # falsecount = 0
        for indices in all_combinations:
            valid_mask = ((indices >= 0) & (indices < self.grid_size)).all(dim=-1)
            # falsecount += torch.sum(valid_mask == False)
            # print(f'false_indices: {torch.nonzero(valid_mask == False)}')
            u_values.append(gather_values(valid_mask, indices))
        
        # print(f'false_count: {falsecount}')
        # Compute interpolations
        out = (
            (((u_values[0] * omw[:, 0] + u_values[1] * weights[:, 0]) * omw[:, 1] + 
            (u_values[2] * omw[:, 0] + u_values[3] * weights[:, 0]) * weights[:, 1]) * omw[:, 2] + 
            ((u_values[4] * omw[:, 0] + u_values[5] * weights[:, 0]) * omw[:, 1] + 
            (u_values[6] * omw[:, 0] + u_values[7] * weights[:, 0]) * weights[:, 1]) * weights[:, 2]) * omw[:, 3] + 
            (((u_values[8] * omw[:, 0] + u_values[9] * weights[:, 0]) * omw[:, 1] + 
            (u_values[10] * omw[:, 0] + u_values[11] * weights[:, 0]) * weights[:, 1]) * omw[:, 2] + 
            ((u_values[12] * omw[:, 0] + u_values[13] * weights[:, 0]) * omw[:, 1] + 
            (u_values[14] * omw[:, 0] + u_values[15] * weights[:, 0]) * weights[:, 1]) * weights[:, 2]) * weights[:, 3]
        )
        out = out.T
        return out
        
    # def total_variation_add_grad(self, wx, wy, wz, wt, dense_mode):
    #     '''Add gradients by total variation loss in-place'''
    #     total_variation_cuda.total_variation_add_grad(
    #         self.grid, self.grid.grad, wx, wy, wz, wt, dense_mode)
        
    def scale_volume_grid(self, uvst_new_world_size):
        def interpolation_4d(data, old_i, old_j, old_k, old_l):
            channels, u, v, s, t = data.shape
            i0, j0, k0, l0 = old_i.floor().long(), old_j.floor().long(), old_k.floor().long(), old_l.floor().long()
            i1, j1, k1, l1 = (i0 + 1).clamp(max=u-1), (j0 + 1).clamp(max=v-1), (k0 + 1).clamp(max=s-1), (l0 + 1).clamp(max=t-1)

            a, b, c, d = old_i - i0, old_j - j0, old_k - k0, old_l - l0

            u0000 = data[:, i0, j0, k0, l0]
            u0001 = data[:, i0, j0, k0, l1]
            u0010 = data[:, i0, j0, k1, l0]
            u0011 = data[:, i0, j0, k1, l1]
            u0100 = data[:, i0, j1, k0, l0]
            u0101 = data[:, i0, j1, k0, l1]
            u0110 = data[:, i0, j1, k1, l0]
            u0111 = data[:, i0, j1, k1, l1]
            u1000 = data[:, i1, j0, k0, l0]
            u1001 = data[:, i1, j0, k0, l1]
            u1010 = data[:, i1, j0, k1, l0]
            u1011 = data[:, i1, j0, k1, l1]
            u1100 = data[:, i1, j1, k0, l0]
            u1101 = data[:, i1, j1, k0, l1]
            u1110 = data[:, i1, j1, k1, l0]
            u1111 = data[:, i1, j1, k1, l1]

            u000 = u0000 * (1 - d) + u0001 * d
            u001 = u0010 * (1 - d) + u0011 * d
            u010 = u0100 * (1 - d) + u0101 * d
            u011 = u0110 * (1 - d) + u0111 * d
            u100 = u1000 * (1 - d) + u1001 * d
            u101 = u1010 * (1 - d) + u1011 * d
            u110 = u1100 * (1 - d) + u1101 * d
            u111 = u1110 * (1 - d) + u1111 * d
            
            u00 = u000 * (1 - c) + u001 * c
            u01 = u010 * (1 - c) + u011 * c
            u10 = u100 * (1 - c) + u101 * c
            u11 = u110 * (1 - c) + u111 * c
            
            u0 = u00 * (1 - b) + u01 * b
            u1 = u10 * (1 - b) + u11 * b
            
            interpolated = u0 * (1 - a) + u1 * a
            return interpolated

        data = self.grid.data
        # batch_size, channels, u, v, s, t = data.shape
        channels, u, v, s, t = data.shape
        new_u, new_v, new_s, new_t = uvst_new_world_size
        new_data = torch.empty([channels, new_u, new_v, new_s, new_t], device=data.device)

        scale_u, scale_v, scale_s, scale_t = new_u / u, new_v / v, new_s / s, new_t / t

        # Create meshgrid for the new indices
        i = torch.arange(new_u, device=data.device).float() / scale_u
        j = torch.arange(new_v, device=data.device).float() / scale_v
        k = torch.arange(new_s, device=data.device).float() / scale_s
        l = torch.arange(new_t, device=data.device).float() / scale_t

        i, j, k, l = torch.meshgrid(i, j, k, l, indexing='ij')

        # Flatten the meshgrid indices
        i = i.flatten()
        j = j.flatten()
        k = k.flatten()
        l = l.flatten()

        # Interpolate the data
        interpolated = interpolation_4d(data, i, j, k, l)

        # Reshape the interpolated data back to the new grid size
        new_data = interpolated.view(channels, new_u, new_v, new_s, new_t)
        self.grid_size = torch.tensor([new_u, new_v, new_s, new_t], dtype=torch.long)
        self.grid = nn.Parameter(new_data)
            
    def extra_repr(self):
        return f'channels={self.channels}, grid_size={self.grid_size.tolist()}'

class LF4DGrid_ONNXSafe(nn.Module):
    def __init__(self, channels, ray_min, ray_max, grid_size, **kwargs):
        super().__init__()
        self.channels = channels
        # self.grid_size = torch.tensor(grid_size, dtype=torch.long)
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.long))
        self.register_buffer('ray_min', torch.tensor(ray_min, dtype=torch.float32))
        self.register_buffer('ray_max', torch.tensor(ray_max, dtype=torch.float32))
        self.grid = nn.Parameter(torch.zeros([channels, *grid_size], dtype=torch.float32))  # [C, U, V, S, T]

    def forward(self, ray):  # ray: [N, 4]
        # 좌표 정규화
        ray_norm = (ray - self.ray_min) / (self.ray_max - self.ray_min)
        grid_idx = ray_norm * (self.grid_size.to(ray.device).float() - 1.0)  # [N, 4]

        # Clamp 경계 처리
        # flatten → clamp per-dimension manually → reconstruct
        max_idx = (self.grid_size - 2).to(ray.device).float()  # [4]
        clamped_split = []
        for d in range(grid_idx.shape[1]):
            clamped_split.append(torch.clamp(grid_idx[:, d], 0.0, max_idx[d]))
        grid_idx = torch.stack(clamped_split, dim=-1)  # [N, 4]

        # 정수 / 소수 가중치 분리
        idx0 = grid_idx.floor().long()  # [N, 4]
        idx1 = idx0 + 1  # [N, 4]
        w = grid_idx - idx0.float()  # [N, 4]
        ow = 1.0 - w

        # 그리드 평탄화 [C, U, V, S, T] → [U*V*S*T, C]
        C, U, V, S, T = self.grid.shape
        flat_grid = self.grid.view(C, -1).permute(1, 0)  # [U*V*S*T, C]

        # 선형 인덱스 4D → 1D
        def lin_idx(ix, iy, iz, iw):
            return ix*(V*S*T) + iy*(S*T) + iz*T + iw  # [N]
        
        # 가중치 합 계산
        out = torch.zeros(ray.shape[0], self.channels, device=ray.device)
        
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                for b2 in [0, 1]:
                    for b3 in [0, 1]:
                        w0 = ow[:, 0] if b0 == 0 else w[:, 0]
                        w1 = ow[:, 1] if b1 == 0 else w[:, 1]
                        w2 = ow[:, 2] if b2 == 0 else w[:, 2]
                        w3 = ow[:, 3] if b3 == 0 else w[:, 3]
                        weight = w0 * w1 * w2 * w3                  # 해당 코너의 전체 가중치 [N]

                        ix = idx0[:, 0] if b0 == 0 else idx1[:, 0]
                        iy = idx0[:, 1] if b1 == 0 else idx1[:, 1]
                        iz = idx0[:, 2] if b2 == 0 else idx1[:, 2]
                        iw = idx0[:, 3] if b3 == 0 else idx1[:, 3]
                        lin = lin_idx(ix, iy, iz, iw)               # 해당 코너의 1D 인덱스 [N]

                        gathered = flat_grid[lin]                   # [N, C]
                        out += gathered * weight.unsqueeze(1)

        return out  # [N, C]

    def extra_repr(self):
        return f'channels={self.channels}, grid_size={self.grid_size.tolist()}'

class LF4DGrid_ONNXSafe_interpolation(nn.Module):
    '''
    interpolation 16, 8, 4, 2, 1 선택 가능
    '''
    def __init__(self, channels, ray_min, ray_max, grid_size,
                 interp_mask=(1,1,1,1),     # interpolation mask
                 add_occ_channel: bool = False,   # 마스크 채널을 끝에 붙일지 여부
                 **kwargs):
        super().__init__()
        self.use_occ = bool(add_occ_channel)
        self.channels = int(channels) + (1 if self.use_occ else 0)  # 채널 수 설정
        self.register_buffer('grid_size', torch.as_tensor(grid_size, dtype=torch.long))
        self.register_buffer('ray_min', torch.as_tensor(ray_min, dtype=torch.float32))
        self.register_buffer('ray_max', torch.as_tensor(ray_max, dtype=torch.float32))
        
        # Parameter grid: [C, U, V, S, T]
        C, U, V, S, T = int(self.channels), int(self.grid_size[0]), int(self.grid_size[1]), int(self.grid_size[2]), int(self.grid_size[3])
        self.grid = nn.Parameter(torch.zeros([C, U, V, S, T], dtype=torch.float32))

        # trace-safe mask
        im_tuple = tuple(bool(int(x)) for x in interp_mask)  # ensure (bool,bool,bool,bool)
        assert len(im_tuple) == 4, "interp_mask must be a 4-tuple/list for (U,V,S,T)."
        self.im0, self.im1, self.im2, self.im3 = im_tuple
        self.register_buffer('interp_mask_tensor', torch.tensor(im_tuple, dtype=torch.bool))

    @torch.no_grad()
    def set_occ_mask(self, mask_4d: torch.Tensor):
        if not self.use_occ:
            raise RuntimeError("This grid was created without add_occ_channel=True.")
        
        while mask_4d.dim() > 4:
            mask_4d = mask_4d.squeeze(0)   # [T,S,V,U]
        
        # list(mask_4d.shape) == [T,S,V,U]인지 확인
        assert list(mask_4d.shape) == [
            self.grid_size[3].item(),
            self.grid_size[2].item(),
            self.grid_size[1].item(),
            self.grid_size[0].item(),
        ]
        occ = mask_4d.permute(3,2,1,0).contiguous()  # [U,V,S,T]

        # grid 마지막 채널을 occ mask (0 or 1)로 덮음
        self.grid.data[-1, ...] = occ.to(device=self.grid.device, dtype=self.grid.dtype)

    def forward(self, ray): 
        ray = ray.to(self.ray_min.dtype)
        ray_norm = (ray - self.ray_min) / (self.ray_max - self.ray_min)  # [N,4]

        grid_size_f = self.grid_size.to(ray.device).float()  # [4]
        grid_idx = ray_norm * (grid_size_f - 1.0)  # [N,4]

        # Per-dimension clamp for interpolated path
        max_idx_interp = (self.grid_size - 2).to(ray.device).float()  # [4]
        grid_idx = torch.minimum(torch.maximum(grid_idx, torch.zeros_like(grid_idx)),
                                 max_idx_interp.view(1, 4).expand_as(grid_idx))

        idx0 = grid_idx.floor().long()  # [N,4]
        idx1 = idx0 + 1
        w    = grid_idx - idx0.float()  # [N,4]
        ow   = 1.0 - w

        # Nearest indices for axes without interpolation
        grid_idx_near = ray_norm * (grid_size_f - 1.0)
        max_idx_near = (self.grid_size - 1).to(ray.device).float()
        grid_idx_near = torch.minimum(torch.maximum(grid_idx_near, torch.zeros_like(grid_idx_near)),
                                      max_idx_near.view(1, 4).expand_as(grid_idx_near))
        nn_idx = grid_idx_near.floor().to(torch.long)

        # Flatten grid to [U*V*S*T, C] to use gather
        C, U, V, S, T = self.grid.shape
        flat_grid = self.grid.view(C, -1).permute(1, 0).contiguous()  # [U*V*S*T, C]

        def lin_idx(ix, iy, iz, iw):
            # U-major linearization for indices in [U,V,S,T]
            return ix*(V*S*T) + iy*(S*T) + iz*T + iw  # [N]

        out = torch.zeros(ray.shape[0], self.channels, device=ray.device, dtype=self.grid.dtype)

        m0, m1, m2, m3 = self.im0, self.im1, self.im2, self.im3  # Python bools (trace-safe)

        # Fixed 16-way loop; combinations disabled by 'active' when axis is nearest and branch chooses idx1
        for b0 in (0, 1):
            for b1 in (0, 1):
                for b2 in (0, 1):
                    for b3 in (0, 1):
                        active = 1.0
                        if (not m0) and b0 == 1: active = 0.0
                        if (not m1) and b1 == 1: active = 0.0
                        if (not m2) and b2 == 1: active = 0.0
                        if (not m3) and b3 == 1: active = 0.0
                        if active == 0.0:
                            continue  # constant-folded at trace-time

                        # Weights per axis
                        w0 = (ow[:, 0] if b0 == 0 else w[:, 0]) if m0 else torch.ones_like(w[:, 0])
                        w1 = (ow[:, 1] if b1 == 0 else w[:, 1]) if m1 else torch.ones_like(w[:, 1])
                        w2 = (ow[:, 2] if b2 == 0 else w[:, 2]) if m2 else torch.ones_like(w[:, 2])
                        w3 = (ow[:, 3] if b3 == 0 else w[:, 3]) if m3 else torch.ones_like(w[:, 3])
                        weight = (w0 * w1 * w2 * w3) * active  # [N]

                        # Indices per axis
                        ix = (idx0[:, 0] if b0 == 0 else idx1[:, 0]) if m0 else nn_idx[:, 0]
                        iy = (idx0[:, 1] if b1 == 0 else idx1[:, 1]) if m1 else nn_idx[:, 1]
                        iz = (idx0[:, 2] if b2 == 0 else idx1[:, 2]) if m2 else nn_idx[:, 2]
                        iw = (idx0[:, 3] if b3 == 0 else idx1[:, 3]) if m3 else nn_idx[:, 3]

                        lin = lin_idx(ix, iy, iz, iw)  # [N]
                        lin = torch.clamp(lin, 0, flat_grid.shape[0] - 1)

                        # ONNX-safe gather
                        lin_expand = lin.unsqueeze(1).expand(-1, self.channels)  # [N, C]
                        gathered = torch.gather(flat_grid, dim=0, index=lin_expand)  # [N, C]

                        out = out + gathered * weight.unsqueeze(1)
        return out  # [N, C]

    def scale_volume_grid(self, new_grid_size_4d):
        raise NotImplementedError("scale_volume_grid is not implemented for LF4DGrid_ONNXSafe_interpolation.")

    def extra_repr(self):
        return f'channels={self.channels}, grid_size={self.grid_size.tolist()}, interp_mask={self.interp_mask_tensor.tolist()}'

''' LF 5D grid
'''
class LF5DGrid(nn.Module):
    def __init__(self, channels, ray_min, ray_max, grid_size, **kwargs):
        super(LF5DGrid, self).__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.grid = nn.Parameter(torch.zeros([1, channels, *grid_size]))
        self.register_buffer('ray_min', torch.Tensor(ray_min))
        self.register_buffer('ray_max', torch.Tensor(ray_max))
        
    def forward(self, ray):
        # nonzero = torch.nonzero(self.lf_grid != 0, as_tuple=False)
        # zero = torch.nonzero(self.lf_grid == 0, as_tuple=False)
        ind_norm = ((ray - self.ray_min) / (self.ray_max - self.ray_min))
        grid_indices = (ind_norm * (self.grid_size - 1))
        output_shape = (self.channels, ray.shape[0])
        out = self.interpolate_5d(grid_indices, output_shape)
        return out
    
    # def interpolate_5d(self, grid_indices, output_shape):
    #     bottom_indices = grid_indices.floor().long()
    #     top_indices = bottom_indices + 1
    #     weights = grid_indices - bottom_indices.float()
    #     ones = torch.ones_like(weights)
    #     omw = ones - weights

    #     def get_valid_values(mask, indices):
    #         result = torch.zeros(output_shape, dtype=self.lf_grid.dtype, device=self.lf_grid.device)
    #         valid_indices = indices[mask]
    #         if valid_indices.size(0) > 0:
    #             result[:, mask] = self.lf_grid[0, :, valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2], valid_indices[:, 3], valid_indices[:, 4]]
    #         return result

    #     valid_mask_b = (bottom_indices >= 0) & (bottom_indices < self.lf_world_size)
    #     valid_mask_t = (top_indices >= 0) & (top_indices < self.lf_world_size)
        
    #     indices = torch.stack([bottom_indices, top_indices], dim=-1)
    #     masks = torch.stack([valid_mask_b, valid_mask_t], dim=-1)
        
    #     u_values = []
    #     for idx_comb in range(32):
    #         mask_comb = (masks[:, :, 0] & masks[:, :, 1])
    #         idx_comb = (indices[:, :, 0] * mask_comb + indices[:, :, 1] * (~mask_comb)).long()
    #         u_values.append(get_valid_values(mask_comb.all(dim=-1), idx_comb))
        
    #     weights_list = [omw, weights]
    #     for dim in range(5):
    #         u_combined = []
    #         for idx_comb in range(0, len(u_values), 2):
    #             u_combined.append(u_values[idx_comb] * weights_list[0][:, dim] + u_values[idx_comb+1] * weights_list[1][:, dim])
    #         u_values = u_combined

    #     out = u_values[0]
    #     out = out.T
    #     return out

    def interpolate_5d(self, grid_indices, output_shape):
        bottom_indices = grid_indices.floor().long()
        top_indices = bottom_indices + 1
        weights = grid_indices - bottom_indices.float()
        ones = torch.ones_like(weights)
        omw = ones - weights
        def gather_values(mask, indices):
            result = torch.zeros(output_shape, dtype=self.lf_grid.dtype, device=self.lf_grid.device)
            valid_indices = indices[mask]
            if valid_indices.size(0) > 0:
                result[:, mask] = self.lf_grid[0, :, valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2], valid_indices[:, 3], valid_indices[:, 4]]
            return result
        valid_mask_b = (bottom_indices >= 0) & (bottom_indices < self.lf_world_size)
        valid_mask_t = (top_indices >= 0) & (top_indices < self.lf_world_size)
        u00000 = gather_values(valid_mask_b.all(dim=-1), bottom_indices)
        u10000 = gather_values(valid_mask_t[:, 0] & valid_mask_b[:, 1:].all(dim=-1), torch.stack([top_indices[:, 0], bottom_indices[:, 1], bottom_indices[:, 2], bottom_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u01000 = gather_values(valid_mask_b[:, 0] & valid_mask_t[:, 1] & valid_mask_b[:, 2:].all(dim=-1), torch.stack([bottom_indices[:, 0], top_indices[:, 1], bottom_indices[:, 2], bottom_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u11000 = gather_values(valid_mask_t[:, :2].all(dim=-1) & valid_mask_b[:, 2:].all(dim=-1), torch.stack([top_indices[:, 0], top_indices[:, 1], bottom_indices[:, 2], bottom_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u00100 = gather_values(valid_mask_b[:, :2].all(dim=-1) & valid_mask_t[:, 2] & valid_mask_b[:, 3:].all(dim=-1), torch.stack([bottom_indices[:, 0], bottom_indices[:, 1], top_indices[:, 2], bottom_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u10100 = gather_values(valid_mask_t[:, [0, 2]].all(dim=-1) & valid_mask_b[:, [1, 3]].all(dim=-1), torch.stack([top_indices[:, 0], bottom_indices[:, 1], top_indices[:, 2], bottom_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u01100 = gather_values(valid_mask_b[:, 0] & valid_mask_t[:, [1, 2]].all(dim=-1) & valid_mask_b[:, 3:].all(dim=-1), torch.stack([bottom_indices[:, 0], top_indices[:, 1], top_indices[:, 2], bottom_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u11100 = gather_values(valid_mask_t[:, :3].all(dim=-1) & valid_mask_b[:, 3:].all(dim=-1), torch.stack([top_indices[:, 0], top_indices[:, 1], top_indices[:, 2], bottom_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u00010 = gather_values(valid_mask_b[:, :3].all(dim=-1) & valid_mask_t[:, 3] & valid_mask_b[:, 4], torch.stack([bottom_indices[:, 0], bottom_indices[:, 1], bottom_indices[:, 2], top_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u10010 = gather_values(valid_mask_t[:, [0, 3]].all(dim=-1) & valid_mask_b[:, 1:3].all(dim=-1) & valid_mask_b[:, 4], torch.stack([top_indices[:, 0], bottom_indices[:, 1], bottom_indices[:, 2], top_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u01010 = gather_values(valid_mask_b[:, 0] & valid_mask_t[:, [1, 3]].all(dim=-1) & valid_mask_b[:, 2:4].all(dim=-1), torch.stack([bottom_indices[:, 0], top_indices[:, 1], bottom_indices[:, 2], top_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u11010 = gather_values(valid_mask_t[:, [0, 1, 3]].all(dim=-1) & valid_mask_b[:, 2:4].all(dim=-1), torch.stack([top_indices[:, 0], top_indices[:, 1], bottom_indices[:, 2], top_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u00110 = gather_values(valid_mask_b[:, :2].all(dim=-1) & valid_mask_t[:, [2, 3]].all(dim=-1) & valid_mask_b[:, 4], torch.stack([bottom_indices[:, 0], bottom_indices[:, 1], top_indices[:, 2], top_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u10110 = gather_values(valid_mask_t[:, [0, 2, 3]].all(dim=-1) & valid_mask_b[:, 1].all(dim=-1) & valid_mask_b[:, 4], torch.stack([top_indices[:, 0], bottom_indices[:, 1], top_indices[:, 2], top_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u01110 = gather_values(valid_mask_b[:, 0] & valid_mask_t[:, [1, 2, 3]].all(dim=-1) & valid_mask_b[:, 4], torch.stack([bottom_indices[:, 0], top_indices[:, 1], top_indices[:, 2], top_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u11110 = gather_values(valid_mask_t[:, :4].all(dim=-1) & valid_mask_b[:, 4], torch.stack([top_indices[:, 0], top_indices[:, 1], top_indices[:, 2], top_indices[:, 3], bottom_indices[:, 4]], dim=1))
        u00001 = gather_values(valid_mask_b[:, :4].all(dim=-1) & valid_mask_t[:, 4], torch.stack([bottom_indices[:, 0], bottom_indices[:, 1], bottom_indices[:, 2], bottom_indices[:, 3], top_indices[:, 4]], dim=1))
        u10001 = gather_values(valid_mask_t[:, [0, 4]].all(dim=-1) & valid_mask_b[:, 1:4].all(dim=-1), torch.stack([top_indices[:, 0], bottom_indices[:, 1], bottom_indices[:, 2], bottom_indices[:, 3], top_indices[:, 4]], dim=1))
        u01001 = gather_values(valid_mask_b[:, 0] & valid_mask_t[:, [1, 4]].all(dim=-1) & valid_mask_b[:, 2:4].all(dim=-1), torch.stack([bottom_indices[:, 0], top_indices[:, 1], bottom_indices[:, 2], bottom_indices[:, 3], top_indices[:, 4]], dim=1))
        u11001 = gather_values(valid_mask_t[:, [0, 1, 4]].all(dim=-1) & valid_mask_b[:, 2:4].all(dim=-1), torch.stack([top_indices[:, 0], top_indices[:, 1], bottom_indices[:, 2], bottom_indices[:, 3], top_indices[:, 4]], dim=1))
        u00101 = gather_values(valid_mask_b[:, :3].all(dim=-1) & valid_mask_t[:, [2, 4]].all(dim=-1) & valid_mask_b[:, 4], torch.stack([bottom_indices[:, 0], bottom_indices[:, 1], top_indices[:, 2], bottom_indices[:, 3], top_indices[:, 4]], dim=1))
        u10101 = gather_values(valid_mask_t[:, [0, 2, 4]].all(dim=-1) & valid_mask_b[:, 1].all(dim=-1) & valid_mask_b[:, 4], torch.stack([top_indices[:, 0], bottom_indices[:, 1], top_indices[:, 2], bottom_indices[:, 3], top_indices[:, 4]], dim=1))
        u01101 = gather_values(valid_mask_b[:, 0] & valid_mask_t[:, [1, 2, 4]].all(dim=-1) & valid_mask_b[:, 4], torch.stack([bottom_indices[:, 0], top_indices[:, 1], top_indices[:, 2], bottom_indices[:, 3], top_indices[:, 4]], dim=1))
        u11101 = gather_values(valid_mask_t[:, [0, 1, 2, 4]].all(dim=-1) & valid_mask_b[:, 4], torch.stack([top_indices[:, 0], top_indices[:, 1], top_indices[:, 2], bottom_indices[:, 3], top_indices[:, 4]], dim=1))
        u00011 = gather_values(valid_mask_b[:, :3].all(dim=-1) & valid_mask_t[:, [3, 4]].all(dim=-1), torch.stack([bottom_indices[:, 0], bottom_indices[:, 1], bottom_indices[:, 2], top_indices[:, 3], top_indices[:, 4]], dim=1))
        u10011 = gather_values(valid_mask_t[:, [0, 3, 4]].all(dim=-1) & valid_mask_b[:, 1:3].all(dim=-1), torch.stack([top_indices[:, 0], bottom_indices[:, 1], bottom_indices[:, 2], top_indices[:, 3], top_indices[:, 4]], dim=1))
        u01011 = gather_values(valid_mask_b[:, 0] & valid_mask_t[:, [1, 3, 4]].all(dim=-1) & valid_mask_b[:, 2], torch.stack([bottom_indices[:, 0], top_indices[:, 1], bottom_indices[:, 2], top_indices[:, 3], top_indices[:, 4]], dim=1))
        u11011 = gather_values(valid_mask_t[:, [0, 1, 3, 4]].all(dim=-1) & valid_mask_b[:, 2], torch.stack([top_indices[:, 0], top_indices[:, 1], bottom_indices[:, 2], top_indices[:, 3], top_indices[:, 4]], dim=1))
        u00111 = gather_values(valid_mask_b[:, :2].all(dim=-1) & valid_mask_t[:, [2, 3, 4]].all(dim=-1), torch.stack([bottom_indices[:, 0], bottom_indices[:, 1], top_indices[:, 2], top_indices[:, 3], top_indices[:, 4]], dim=1))
        u10111 = gather_values(valid_mask_t[:, [0, 2, 3, 4]].all(dim=-1) & valid_mask_b[:, 1], torch.stack([top_indices[:, 0], bottom_indices[:, 1], top_indices[:, 2], top_indices[:, 3], top_indices[:, 4]], dim=1))
        u01111 = gather_values(valid_mask_b[:, 0] & valid_mask_t[:, [1, 2, 3, 4]].all(dim=-1), torch.stack([bottom_indices[:, 0], top_indices[:, 1], top_indices[:, 2], top_indices[:, 3], top_indices[:, 4]], dim=1))
        u11111 = gather_values(valid_mask_t.all(dim=-1), top_indices)
        
        u0000 = u00000 * omw[:, 0] + u10000 * weights[:, 0]
        u0001 = u00001 * omw[:, 0] + u10001 * weights[:, 0]
        u0010 = u00010 * omw[:, 0] + u10010 * weights[:, 0]
        u0011 = u00011 * omw[:, 0] + u10011 * weights[:, 0]
        u0100 = u00100 * omw[:, 0] + u10100 * weights[:, 0]
        u0101 = u00101 * omw[:, 0] + u10101 * weights[:, 0]
        u0110 = u00110 * omw[:, 0] + u10110 * weights[:, 0]
        u0111 = u00111 * omw[:, 0] + u10111 * weights[:, 0]
        u1000 = u01000 * omw[:, 0] + u11000 * weights[:, 0]
        u1001 = u01001 * omw[:, 0] + u11001 * weights[:, 0]
        u1010 = u01010 * omw[:, 0] + u11010 * weights[:, 0]
        u1011 = u01011 * omw[:, 0] + u11011 * weights[:, 0]
        u1100 = u01100 * omw[:, 0] + u11100 * weights[:, 0]
        u1101 = u01101 * omw[:, 0] + u11101 * weights[:, 0]
        u1110 = u01110 * omw[:, 0] + u11110 * weights[:, 0]
        u1111 = u01111 * omw[:, 0] + u11111 * weights[:, 0]
        
        u000 = u0000 * omw[:, 1] + u1000 * weights[:, 1]
        u001 = u0001 * omw[:, 1] + u1001 * weights[:, 1]
        u010 = u0010 * omw[:, 1] + u1010 * weights[:, 1]
        u011 = u0011 * omw[:, 1] + u1011 * weights[:, 1]
        u100 = u0100 * omw[:, 1] + u1100 * weights[:, 1]
        u101 = u0101 * omw[:, 1] + u1101 * weights[:, 1]
        u110 = u0110 * omw[:, 1] + u1110 * weights[:, 1]
        u111 = u0111 * omw[:, 1] + u1111 * weights[:, 1]
        
        u00 = u000 * omw[:, 2] + u100 * weights[:, 2]
        u01 = u001 * omw[:, 2] + u101 * weights[:, 2]
        u10 = u010 * omw[:, 2] + u110 * weights[:, 2]
        u11 = u011 * omw[:, 2] + u111 * weights[:, 2]
        
        u0 = u00 * omw[:, 3] + u10 * weights[:, 3]
        u1 = u01 * omw[:, 3] + u11 * weights[:, 3]
        
        out = u0 * omw[:, 4] + u1 * weights[:, 4]
        out = out.T
        return out
    
    
    def scale_volume_grid(self, new_lf_world_size):
        def interpolation_5d(data, old_i, old_j, old_k, old_l, old_m):
            batch_size, channels, u, v, s, t, w = data.shape
            i0, j0, k0, l0, m0 = old_i.floor().long(), old_j.floor().long(), old_k.floor().long(), old_l.floor().long(), old_m.floor().long()
            i1, j1, k1, l1, m1 = (i0 + 1).clamp(max=u-1), (j0 + 1).clamp(max=v-1), (k0 + 1).clamp(max=s-1), (l0 + 1).clamp(max=t-1), (m0 + 1).clamp(max=w-1)

            a, b, c, d, e = old_i - i0, old_j - j0, old_k - k0, old_l - l0, old_m - m0

            u00000 = data[:, :, i0, j0, k0, l0, m0]
            u00001 = data[:, :, i0, j0, k0, l0, m1]
            u00010 = data[:, :, i0, j0, k0, l1, m0]
            u00011 = data[:, :, i0, j0, k0, l1, m1]
            u00100 = data[:, :, i0, j0, k1, l0, m0]
            u00101 = data[:, :, i0, j0, k1, l0, m1]
            u00110 = data[:, :, i0, j0, k1, l1, m0]
            u00111 = data[:, :, i0, j0, k1, l1, m1]
            u01000 = data[:, :, i0, j1, k0, l0, m0]
            u01001 = data[:, :, i0, j1, k0, l0, m1]
            u01010 = data[:, :, i0, j1, k0, l1, m0]
            u01011 = data[:, :, i0, j1, k0, l1, m1]
            u01100 = data[:, :, i0, j1, k1, l0, m0]
            u01101 = data[:, :, i0, j1, k1, l0, m1]
            u01110 = data[:, :, i0, j1, k1, l1, m0]
            u01111 = data[:, :, i0, j1, k1, l1, m1]
            u10000 = data[:, :, i1, j0, k0, l0, m0]
            u10001 = data[:, :, i1, j0, k0, l0, m1]
            u10010 = data[:, :, i1, j0, k0, l1, m0]
            u10011 = data[:, :, i1, j0, k0, l1, m1]
            u10100 = data[:, :, i1, j0, k1, l0, m0]
            u10101 = data[:, :, i1, j0, k1, l0, m1]
            u10110 = data[:, :, i1, j0, k1, l1, m0]
            u10111 = data[:, :, i1, j0, k1, l1, m1]
            u11000 = data[:, :, i1, j1, k0, l0, m0]
            u11001 = data[:, :, i1, j1, k0, l0, m1]
            u11010 = data[:, :, i1, j1, k0, l1, m0]
            u11011 = data[:, :, i1, j1, k0, l1, m1]
            u11100 = data[:, :, i1, j1, k1, l0, m0]
            u11101 = data[:, :, i1, j1, k1, l0, m1]
            u11110 = data[:, :, i1, j1, k1, l1, m0]
            u11111 = data[:, :, i1, j1, k1, l1, m1]
            
            u0000 = u00000 * (1 - e) + u00001 * e
            del u00000, u00001
            u0001 = u00010 * (1 - e) + u00011 * e
            del u00010, u00011
            u0010 = u00100 * (1 - e) + u00101 * e
            del u00100, u00101
            u0011 = u00110 * (1 - e) + u00111 * e
            del u00110, u00111
            u0100 = u01000 * (1 - e) + u01001 * e
            del u01000, u01001
            u0101 = u01010 * (1 - e) + u01011 * e
            del u01010, u01011
            u0110 = u01100 * (1 - e) + u01101 * e
            del u01100, u01101
            u0111 = u01110 * (1 - e) + u01111 * e
            del u01110, u01111
            u1000 = u10000 * (1 - e) + u10001 * e
            del u10000, u10001
            u1001 = u10010 * (1 - e) + u10011 * e
            del u10010, u10011
            u1010 = u10100 * (1 - e) + u10101 * e
            del u10100, u10101
            u1011 = u10110 * (1 - e) + u10111 * e
            del u10110, u10111
            u1100 = u11000 * (1 - e) + u11001 * e
            del u11000, u11001
            u1101 = u11010 * (1 - e) + u11011 * e
            del u11010, u11011
            u1110 = u11100 * (1 - e) + u11101 * e
            del u11100, u11101
            u1111 = u11110 * (1 - e) + u11111 * e
            del u11110, u11111
            u000 = u0000 * (1 - d) + u0001 * d
            u001 = u0010 * (1 - d) + u0011 * d
            u010 = u0100 * (1 - d) + u0101 * d
            u011 = u0110 * (1 - d) + u0111 * d
            u100 = u1000 * (1 - d) + u1001 * d
            u101 = u1010 * (1 - d) + u1011 * d
            u110 = u1100 * (1 - d) + u1101 * d
            u111 = u1110 * (1 - d) + u1111 * d
            
            u00 = u000 * (1 - c) + u001 * c
            u01 = u010 * (1 - c) + u011 * c
            u10 = u100 * (1 - c) + u101 * c
            u11 = u110 * (1 - c) + u111 * c
            
            u0 = u00 * (1 - b) + u01 * b
            u1 = u10 * (1 - b) + u11 * b
            
            interpolated = u0 * (1 - a) + u1 * a
            return interpolated
        
        data = self.lf_grid.data
        batch_size, channels, x, z, dx, dy, dz = data.shape
        new_x, new_z, new_dx, new_dy, new_dz = new_lf_world_size
        new_data = torch.zeros([batch_size, channels, *new_lf_world_size], dtype=data.dtype, device=data.device)
        
        scale_x, scale_z, scale_dx, scale_dy, scale_dz = new_x / x, new_z / z, new_dx / dx, new_dy / dy, new_dz / dz
        
        # Create meshgrid for the new indices
        i = torch.arange(new_x, device=data.device).float() / scale_x
        j = torch.arange(new_z, device=data.device).float() / scale_z
        k = torch.arange(new_dx, device=data.device).float() / scale_dx
        l = torch.arange(new_dy, device=data.device).float() / scale_dy
        m = torch.arange(new_dz, device=data.device).float() / scale_dz
        
        i, j, k, l, m = torch.meshgrid(i, j, k, l, m, indexing='ij')
        
        # Flatten the meshgrid indices
        i, j, k, l, m = i.flatten(), j.flatten(), k.flatten(), l.flatten(), m.flatten()
        
        # Interpolate the data
        interpolated = interpolation_5d(data, i, j, k, l, m)
        
        # Reshape the interpolated data back to the new grid size
        new_data = interpolated.view(batch_size, channels, *new_lf_world_size)
        self.lf_world_size = torch.tensor([new_x, new_z, new_dx, new_dy, new_dz], dtype=torch.long)
        self.lf_grid = nn.Parameter(new_data)
            
    def extra_repr(self):
        return f'channels={self.channels}, lf_world_size={self.lf_world_size.tolist()}'
