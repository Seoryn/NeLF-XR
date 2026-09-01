import os, time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import vggrid
# from .gridencoder import GridEncoder

'''Twoplane Model'''
class DLFGO_twoplane(torch.nn.Module):
    def __init__(self, ray_min, ray_max, grid_num, ray_type='twoplane',
                 grid_dim=16, mlp_depth=2, mlp_width=128, pe=0,
                 decomp='4d', levels=[1],
                 interp_mode=16,
                 use_occ_mask=False,
                 occ_tau=0.2,
                 bg_val=0.0,
                 occ_reduce='max'):
        super(DLFGO_twoplane, self).__init__()
        self.register_buffer('ray_min', torch.Tensor(ray_min))
        self.register_buffer('ray_max', torch.Tensor(ray_max))
        self.grid_num = grid_num
        self._set_grid_resolution(grid_num)
        self.grid_dim = grid_dim
        self.mlp_kwargs = {
            'mlp_depth': mlp_depth, 'mlp_width': mlp_width,
        }
        self.pe = pe
        self.decomp = decomp
        self.levels = levels
        if self.levels == 1:
            self.level = [1]
        if self.levels == 4:
            self.level = [1, 2, 4, 8]
        if self.levels == 8:
            self.level = [1, 2, 3, 4, 5, 6, 7, 8]
        if self.levels == 16:
            self.level = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        if self.decomp == '1d':
            self.grid_type = 'LF1DGrid'
            self.grid_combinations = [
                [0], [1], [2], [3],
            ]
        if self.decomp == '2d':
            self.grid_type = 'LF2DGrid'
            self.grid_combinations = [
                [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],   # 6개
            ]
        if self.decomp == '3d':
            self.grid_type = 'LF3DGrid'
            self.grid_combinations = [
                [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],    # 4개
            ]
        if self.decomp == '4d':
            self.grid_type = 'LF4DGrid'
            self.grid_combinations = [
                [0, 1, 2, 3],   # 1개
            ]
        self.interp_mode = interp_mode
        self.use_occ_mask = bool(use_occ_mask and decomp=='4d')
        self.occ_tau = float(occ_tau)
        self.bg_val = float(bg_val) if not isinstance(bg_val, (tuple,list)) else torch.tensor(bg_val, dtype=torch.float32)
        self.occ_reduce = occ_reduce
        mask_map = {
            16: (1,1,1,1),
            8: (1,1,1,0),
            4: (1,1,0,0),
            2: (1,0,0,0),
            1: (0,0,0,0),
        }
        self.interp_mask = mask_map.get(int(interp_mode), (1,1,1,1))
        self.grid = nn.ModuleList()
        for i, level in enumerate(self.level):
            grid_modules = nn.ModuleList([
                vggrid.create_grid(
                    type=self.grid_type,
                    # channels=self.grid_dim + (1 if (self.decomp=='4d' and self.use_occ_mask) else 0),
                    channels=self.grid_dim,
                    ray_min=self.ray_min[grid_comb],
                    ray_max=self.ray_max[grid_comb],
                    grid_size=torch.tensor(
                        (self.grid_size[grid_comb] * level).tolist(),
                         dtype=torch.long
                    ),
                    **({'interp_mask': self.interp_mask} if self.decomp == '4d' else {})
                ) for grid_comb in self.grid_combinations
            ])
            self.grid.append(grid_modules)

        # occ mask 채널 제외하고 feature만
        per_grid_feat = self.grid_dim
        mlp_dim = per_grid_feat * len(self.grid_combinations) * self.levels
        print('dlfgo_twoplane: grid', self.grid)
        # if self.pe > 0:
        #     self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(self.pe)]))
        #     mlp_dim += (3+3*pe*2)
        # self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(self.pe)]))
        # mlp_dim += (3+3*pe*2)
        # self.encoder = GridEncoder(input_dim=4, num_levels=8, level_dim=2, base_resolution=16, log2_hashmap_size=24, desired_resolution=512, gridtype='hash', align_corners=True)
        # print('dlfgo_twoplane: hash encoder', self.encoder)
        # mlp_dim = self.encoder.output_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim, mlp_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(mlp_width, mlp_width), nn.ReLU(inplace=True))
                for _ in range(mlp_depth-2)
            ],
            nn.Linear(mlp_width, 3),
        )
        nn.init.constant_(self.mlp[-1].bias, 0)
        print('dlfgo_twoplane: mlp', self.mlp)
            
    def _set_grid_resolution(self, grid_num):
        self.grid_size = torch.tensor(grid_num, dtype=torch.int)
        print('dlfgo_twoplane: grid_size ', self.grid_size)

    def get_kwargs(self):
        return {
            'ray_min': self.ray_min.cpu().numpy(),
            'ray_max': self.ray_max.cpu().numpy(),
            'grid_num': self.grid_num,
            'grid_dim': self.grid_dim,
            **self.mlp_kwargs,
            'pe': self.pe,
            'decomp': self.decomp,
            'levels': self.levels,
            'interp_mode': int(self.interp_mode)
        }
    
    @torch.no_grad()
    def scale_volume_grid(self, factor=1.2):
        print('dlfgo_twoplane: scale_volume_grid start')
        ori_grid_size = self.grid_size.tolist()
        new_grid_size = self.grid_size * factor
        self._set_grid_resolution(new_grid_size)
        print('dlfgo_twoplane: scale_volume_grid scale world_size from', ori_grid_size, 'to', self.grid_size)
        for i in range(self.levels):
            for j, grid_comb in enumerate(self.grid_combinations):
                self.grid[i][j].scale_volume_grid(new_grid_size[grid_comb])
        
    @torch.no_grad()
    def set_occ_mask_4d(self, mask_4d):
        """
        mask_4d: [T, S, V, U]
        grid: [C, U, V, S, T]
        """
        if not self.use_occ_mask:
            print("[WARN] use_occ_mask=False or not 4D; set_occ_mask_4d skipped.")
            return
        
        print(f"[INFO] Setting occ mask: {mask_4d.shape}")

        for i in range(self.levels):
            for j, grid_comb in enumerate(self.grid_combinations):
                g = self.grid[i][j]
                
                # Grid 현재 shape 확인
                if hasattr(g, 'grid'):
                    print(f"  Level {i}, Grid {j}: grid.shape = {g.grid.shape}")
                    
                    # 마스크를 grid 좌표계에 맞게 permute: [T,S,V,U] → [U,V,S,T]
                    mask_permuted = mask_4d.permute(3, 2, 1, 0)  # [U,V,S,T]
                    
                    # Grid 크기에 맞게 interpolate (multi-level 지원)
                    grid_spatial_size = g.grid.shape[1:]  # (U, V, S, T)
                    if mask_permuted.shape != grid_spatial_size:
                        mask_permuted = mask_permuted.unsqueeze(0).unsqueeze(0)  # [1,1,U,V,S,T]
                        mask_permuted = F.interpolate(
                            mask_permuted.view(1, 1, *mask_permuted.shape[2:]),
                            size=grid_spatial_size,
                            mode='nearest'
                        ).squeeze(0).squeeze(0)
                    
                    mask_channel = mask_permuted.unsqueeze(0)  # [1, U, V, S, T]
                    
                    # 마지막 채널에 mask 설정
                    g.grid.data[-1:] = mask_channel.to(g.grid.device, g.grid.dtype)
                    
                    print(f"  → Mask set to channel {g.grid.shape[0]-1}")
                    print(f"  → Mask range: [{g.grid.data[-1].min():.3f}, {g.grid.data[-1].max():.3f}]")
    
    def forward(self, ray, global_step=None):
        per_grid_dim = (self.grid_dim + 1) if (self.decomp=='4d' and self.use_occ_mask) else self.grid_dim
        chunks = []
        for i in range(self.levels):
            for j, grid_comb in enumerate(self.grid_combinations):
                f_grid = self.grid[i][j](ray[:, grid_comb])
                if global_step == 'debug' and i == 0 and j == 0:
                    np.save('torch_fgrid.npy', f_grid.detach().cpu().numpy()) # npy 저장
                chunks.append(f_grid)

        if self.decomp=='4d' and self.use_occ_mask:
            # grid에서 feature / occ 채널 분리
            grids_out = torch.stack(chunks, dim=0)
            feat = grids_out[..., :self.grid_dim]                   # [G, N, grid_dim]
            occ  = grids_out[..., self.grid_dim:self.grid_dim+1]    # [G, N, 1]

            # 추가; 디버깅
            print(f"\n[Forward Debug]")
            print(f"  grids_out shape: {grids_out.shape}")
            print(f"  feat shape: {feat.shape}, range: [{feat.min():.3f}, {feat.max():.3f}]")
            print(f"  occ shape: {occ.shape}, range: [{occ.min():.3f}, {occ.max():.3f}]")
            print(f"  occ > {self.occ_tau}: {(occ > self.occ_tau).sum()} / {occ.numel()}")

            feat_cat = feat.permute(1,0,2).contiguous().view(ray.shape[0], -1)  # [N, grid_dim*G]
            
            # 여러 grid의 마스크를 픽셀 단위로 하나의 값으로 합침
            if self.occ_reduce == 'mean':
                occ_agg = occ.mean(dim=0)   # [N,1]
            else:
                occ_agg = occ.max(dim=0).values   # [N,1]

            # 집계된 마스크 이진화 (0 또는 1)
            active = (occ_agg > self.occ_tau).to(feat_cat.dtype)  # [N,1]

            # 전체에 대해 MLP + sigmoid 실행
            y_full = self.mlp(feat_cat)     # [N, 3]
            y_full = torch.sigmoid(y_full)  # [N, 3]

            # 배경값 생성 (self.bg_val을 텐서로)
            if torch.is_tensor(self.bg_val):
                bg = self.bg_val.to(y_full.device, dtype=y_full.dtype).view(1, -1)
                if bg.shape[1] == 1:
                    bg = bg.expand(ray.shape[0], 3)     # [N, 3]
                elif bg.shape[1] == 3:
                    bg = bg.expand(ray.shape[0], -1)    # [N, 3]
            else:
                bg = torch.full((ray.shape[0], 3), self.bg_val, 
                            device=y_full.device, dtype=y_full.dtype)

            # active mask 적용: 전경이면 y_all, 배경이면 bg
            active_3ch = active.expand(-1, 3)  # [N,1] → [N,3]
            y = active_3ch * y_full + (1.0 - active_3ch) * bg
            
            if global_step == 'debug':
                print(f"  active pixels: {active.sum().item()} / {active.shape[0]}")
                print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")
                print(f"  bg value: {bg[0].tolist()}")

            return y

        results = torch.cat(chunks, dim=-1)
        results = self.mlp(results)
        render_result = torch.sigmoid(results)
        return render_result