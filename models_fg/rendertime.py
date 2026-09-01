import argparse, time, statistics as stats
import numpy as np
import onnxruntime as ort
import torch
import imageio
import re
import os
from glob import glob
from pathlib import Path

H, W = 512, 512
# H, W = 384, 384
# H, W = 256, 256
BATCH = H * W

def now():
    return time.perf_counter()

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def load_gt_images(image_dir, v_size=17, u_size=17):
    pattern = re.compile(r'out_(\d+)_(\d+)_.*\.png')
    image_files = sorted(glob(str(Path(image_dir) / 'out_*.png')))

    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {image_dir}")
    
    # 첫 파일로 크기 확인
    first_img = imageio.v2.imread(image_files[0])
    h, w = first_img.shape[:2]

    gt_images = np.zeros((v_size, u_size, h, w, 3), dtype=np.float32)

    loaded = 0
    for img_path in image_files:
        name = Path(img_path).name
        match = pattern.match(name)

        v_idx = int(match.group(1))
        u_idx = int(match.group(2))

        img = imageio.v2.imread(img_path)

        # grayscale → RGB
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        # [0, 1]
        img = img.astype(np.float32) / 255.0
        gt_images[v_idx, u_idx] = img
        loaded += 1

    print(f"Loaded {loaded} GT images (expected {v_size * u_size})")
    
    # flatten
    gt_flat = gt_images.reshape(-1, 3)  # [V, U, H, W, 3] → [V*U*H*W, 3]
    
    return gt_flat

# flat index → (v, u, y, x)
def flat_idx_to_coords(flat_indices, v_size, u_size, h, w):
    pixels_per_view = h * w             # 512*512
    pixels_per_v_row = u_size * h * w   # 17*(512*512)

    v = flat_indices // pixels_per_v_row
    remainder = flat_indices % pixels_per_v_row

    u = remainder // pixels_per_view
    remainder = remainder % pixels_per_view

    y = remainder // w
    x = remainder % w

    return v, u, y, x

# (v, u, y, x) → LF ray
def coords_to_rays(v, u, y, x, v_size, u_size, h, w, c2w_grid):
    uv_scale = 1.0
    st_scale = 0.25

    u_ray = ((x + 0.5) / w * 2.0 - 1.0) * uv_scale          # [-1, 1]
    v_ray = (1.0 - (y + 0.5) / h * 2.0) * uv_scale          # [1, -1]

    rays_list = []
    for i in range(len(v)):
        v_idx = v[i]
        u_idx = u[i]
        c2w = c2w_grid[v_idx, u_idx]    # [4, 4]

        s_ray = c2w[0, 3] * st_scale
        t_ray = c2w[1, 3] * st_scale

        rays_list.append([s_ray, t_ray, u_ray[i], v_ray[i]])
    
    rays = np.array(rays_list, dtype=np.float32)
    return rays

# ONNX Runtime
def build_session(onnx_path, device="cuda", intra_threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if device == "cuda":
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
                     "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

# n_rays만큼 랜덤 ray 생성
def make_inputs(n_rays, seed=0):
    rng = np.random.default_rng(seed)
    rays = rng.random((n_rays, 4), dtype=np.float32)
    return rays

class FGRenderer:
    def __init__(self, sess, mask_path=None):
        self.sess = sess
        
        # 마스크 로드
        if mask_path is not None:
            mask_2d = torch.load(mask_path, map_location='cpu')
            mask_flat = (mask_2d.flatten() > 0.5).numpy().astype(bool)

            self.fg_indices = np.where(mask_flat)[0]    # 전경 인덱스
            self.n_total = len(mask_flat)
            self.n_fg = len(self.fg_indices)

            print(f"[FG] FG pixels: {self.n_fg} ({self.n_fg/self.n_total*100:.1f}%)")
        else:
            self.fg_indices = None
            self.n_total = BATCH
            self.n_fg = BATCH

            print("[FG + BG] Rendering all pixels")

    # FG + BG
    def render_all(self, n_rays, seed):
        rays = make_inputs(n_rays, seed)
        return self.sess.run(None, {"ray": rays})[0]
    
    # FG
    def render_only_fg(self, n_rays, seed):
        rays = make_inputs(n_rays, seed)
        return self.sess.run(None, {"ray": rays})[0]
    
class FGRenderer4D:
    def __init__(self, sess, mask_4d_path=None):
        self.sess = sess
        self.has_mask = False

        if mask_4d_path is not None:
            mask_data = torch.load(mask_4d_path, map_location='cpu')    # 4D 마스크 로드

            if isinstance(mask_data, dict):
                mask_4d = mask_data['occ']                      # (u, v, s, t)
                self.grid_num = mask_data['grid_num']
                self.ray_min = mask_data['ray_min'].numpy()     # [s0, t0, u0, v0]
                self.ray_max = mask_data['ray_max'].numpy()     # [s1, t1, u1, v1]
                # print(f"4D Mask Grid num: {self.grid_num}")
                # print(f"4D Mask Ray min: {self.ray_min}")
                # print(f"4D Mask Ray max: {self.ray_max}")

                Nu, Nv, Ns, Nt = self.grid_num
                print(f"4D Mask shape: {list(mask_4d.shape)} (U, V, S, T)")

                # 픽셀 기반 (17*17*512*512)
                # self.v = self.u = 17
                # self.H = self.W = 512

                occ_np = mask_4d.numpy()
                bu, bv, bs, bt = np.where(occ_np)   # True 인덱스만

                self.fg_total = len(bu)             # 전경 voxel 개수
                print(f"[FG] FG voxels: {self.fg_total:,} / {Nu*Nv*Ns*Nt:,} ({100*self.fg_total/(Nu*Nv*Ns*Nt):.1f}%)")

                # 인덱스 및 마스크 저장
                self.occ_np = occ_np
                self.bu = bu
                self.bv = bv
                self.bs = bs
                self.bt = bt

                # UVST grid → ray 좌표
                s0, t0, u0, v0 = self.ray_min
                s1, t1, u1, v1 = self.ray_max

                u_ray = u0 + (bu + 0.5) / Nu * (u1 - u0)
                v_ray = v0 + (bv + 0.5) / Nv * (v1 - v0)
                s_ray = s0 + (bs + 0.5) / Ns * (s1 - s0)
                t_ray = t0 + (bt + 0.5) / Nt * (t1 - t0)

                self.fg_rays = np.stack([s_ray, t_ray, u_ray, v_ray], axis=1).astype(np.float32)    # [N_fg, 4]
                print(f"[FG] Generated {len(self.fg_rays):,} rays from UVST grid")

                self.has_mask = True
            else:
                # 마스크 파일 형식이 dict이도록
                raise ValueError("mask dict must contain occ/grid_num/ray_min/ray_max")

            if mask_4d.ndim != 4:
                raise ValueError(f"Expected 4D mask, got {mask_4d.ndim}D")  # 4D인지 확인
            
            # # 모든 전경 픽셀을 하나의 배열로 만듦
            # all_fg_indices = []     
            # self.view_starts = []   
            # self.view_counts = []   

            # current_offset = 0
            # pixels = self.H * self.W

            # for v in range(self.v):
            #     for u in range(self.u):
            #         # flat indexing 접근
            #         view_offset = (v * self.u + u) * pixels
            #         view_mask = self.mask_flat[view_offset : view_offset + pixels]

            #         # 전경 픽셀의 로컬 인덱스
            #         local_fg_indices = np.where(view_mask)[0]
            #         n_fg = len(local_fg_indices)

            #         # 글로벌 인덱스로 변환
            #         all_fg_indices.append(local_fg_indices)     # [[uv[0,0]], [uv[0,1]], ..., [uv[16,16]]]
            #         self.view_starts.append(current_offset)     # 각 view의 전경 시작 위치
            #         self.view_counts.append(n_fg)               # 각 view의 전경 픽셀 수

            #         current_offset += n_fg

            # # 하나로 합침
            # self.all_fg_indices = np.concatenate(all_fg_indices)
            
            # total_pixels = self.v * self.u * self.H * self.W
        
        else:
            # 마스크 없어도 UVST 파라미터 설정
            self.grid_num = [8, 8, 64, 32]
            self.ray_min = np.array([-0.25, -0.25, -1.0, -1.0], dtype=np.float32)
            self.ray_max = np.array([ 0.25,  0.25,  1.0,  1.0], dtype=np.float32)
            self.fg_total = 0

            # 픽셀 기반 (17*17*512*512)
            # self.v = self.u = 17
            # self.H = self.W = 512
            # self.fg_total = self.v * self.u * self.H * self.W
            print("[FG + BG] Rendering all rays")

        Nu, Nv, Ns, Nt = self.grid_num
        self.total_pixels = Nu * Nv * Ns * Nt

    def render_all(self, seed=0):
        Nu, Nv, Ns, Nt = self.grid_num
        s0, t0, u0, v0 = self.ray_min
        s1, t1, u1, v1 = self.ray_max
        
        total_voxels = Nu * Nv * Ns * Nt

        # 전체 UVST grid 인덱스
        bu, bv, bs, bt = np.meshgrid(
            np.arange(Nu),
            np.arange(Nv),
            np.arange(Ns),
            np.arange(Nt),
            indexing='ij'
        )

        # [Nu, Nv, Ns, Nt] → [Nu*Nv*Ns*Nt]
        bu = bu.flatten()
        bv = bv.flatten()
        bs = bs.flatten()
        bt = bt.flatten()
        
        u_ray = u0 + (bu + 0.5) / Nu * (u1 - u0)
        v_ray = v0 + (bv + 0.5) / Nv * (v1 - v0)
        s_ray = s0 + (bs + 0.5) / Ns * (s1 - s0)
        t_ray = t0 + (bt + 0.5) / Nt * (t1 - t0)
        
        rays = np.stack([s_ray, t_ray, u_ray, v_ray], axis=1).astype(np.float32)

        # 렌더링
        t0 = now()
        rgb = self.sess.run(None, {"ray": rays})[0]
        t1 = now()

        return t1 - t0, rgb

    def render_only_fg(self, seed=0):        
        rays = self.fg_rays     # UVST grid의 전경 ray 사용
        
        t0 = now()
        rgb = self.sess.run(None, {"ray": rays})[0]
        t1 = now()

        return t1 - t0, rgb

# 추론 + 시간 측정
def run(renderer, n_rays, seed, mode='all'):
    t0 = now()
    if mode == 'all':
        _ = renderer.render_all(n_rays, seed)   # 추론
    else:
        _ = renderer.render_only_fg(n_rays, seed)
    t1 = now()
    return (t1 - t0)                    # infer

# 시간 리스트 요약 통계
def summarize(xs):
    return {
        "mean_ms": 1000*stats.mean(xs),
        "std_ms":  1000*stats.pstdev(xs),
        "p50_ms":  1000*stats.median(xs),
        "p95_ms":  1000*np.percentile(xs, 95),
    }


''' 디버깅용
'''
def debug_sampling(renderer_4d, gt_images_flat, n_samples=10):
    print("\n" + "="*60)
    print("DEBUG: Sampling Check")
    print("="*60)
    
    rng = np.random.default_rng(0)
    sampled_flat_idx = rng.choice(renderer_4d.fg_flat_indices, size=n_samples, replace=False)
    
    # Flat index → (v, u, y, x)
    v, u, y, x = flat_idx_to_coords(
        sampled_flat_idx, renderer_4d.v, renderer_4d.u, renderer_4d.H, renderer_4d.W
    )
    
    # (v, u, y, x) → rays
    rays = coords_to_rays(v, u, y, x,
                          renderer_4d.v, renderer_4d.u, renderer_4d.H, renderer_4d.W,
                          renderer_4d.c2w_grid)
    
    # 렌더링
    rgb_sample = renderer_4d.sess.run(None, {"ray": rays})[0]
    
    # GT
    gt_sample = gt_images_flat[sampled_flat_idx]
    
    # 출력
    print(f"\nSample size: {n_samples}")
    print(f"\nFlat indices (first 5):")
    print(sampled_flat_idx[:5])
    
    print(f"\nCoordinates (first 5):")
    for i in range(min(5, n_samples)):
        print(f"  [{i}] v={v[i]:2d}, u={u[i]:2d}, y={y[i]:3d}, x={x[i]:3d}")
    
    print(f"\nRays (first 3):")
    print(rays[:3])
    
    print(f"\nModel Output RGB (first 3):")
    print(rgb_sample[:3])
    
    print(f"\nGT RGB (first 3):")
    print(gt_sample[:3])
    
    print(f"\nDifference (first 3):")
    print(np.abs(rgb_sample[:3] - gt_sample[:3]))
    
    print(f"\nMSE per sample (first 10):")
    mse_per_sample = np.mean((rgb_sample - gt_sample)**2, axis=1)
    print(mse_per_sample[:10])
    
    print(f"\nOverall MSE: {np.mean(mse_per_sample):.6f}")
    print(f"Overall PSNR: {calculate_psnr(rgb_sample, gt_sample):.2f} dB")
    
    # RGB 범위 체크
    print(f"\nModel RGB range: [{rgb_sample.min():.3f}, {rgb_sample.max():.3f}]")
    print(f"GT RGB range:    [{gt_sample.min():.3f}, {gt_sample.max():.3f}]")
    
    print("="*60 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/seoryn/onnx/dlfgo_onnx/models_fg/bracelet_small_fg_8_8.onnx")
    ap.add_argument("--mask", default=None)
    ap.add_argument("--mode", choices=["2d", "4d"], default='2d')
    ap.add_argument("--render_mode", choices=["all", "fg"], default='all')
    ap.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interp", type=int, default=16)
    args = ap.parse_args()

    sess = build_session(args.model, args.device, args.threads)

    # 2D mask
    if args.mode == '2d':
        renderer = FGRenderer(sess, args.mask)

        # ray 개수 결정
        n_rays_all = renderer.n_total   # 262144
        n_rays_fg = renderer.n_fg       # 106641

        # warm up
        if args.render_mode == 'all':
            for _ in range(args.warmup):
                renderer.render_all(n_rays_all, args.seed)
        else:
            for _ in range(args.warmup):
                renderer.render_only_fg(n_rays_fg, args.seed)

        times = []

        # FG + BG
        if args.render_mode == 'all':
            for i in range(args.iters):
                rays = make_inputs(n_rays_all, args.seed + i)
                t0 = now()
                _ = renderer.sess.run(None, {"ray": rays})[0]
                t1 = now()
                times.append(t1 - t0)
            n_rays = n_rays_all
        # FG
        else:
            for i in range(args.iters):
                rays = make_inputs(n_rays_fg, args.seed + i)
                t0 = now()
                _ = renderer.sess.run(None, {"ray": rays})[0]
                t1 = now()
                times.append(t1 - t0)
            n_rays = n_rays_fg

        s = summarize(times)

        print(f"Resolution: {H}x{W} (batch={BATCH})  Device: {args.device}")
        print(f"Rays:   {n_rays:,}")
        print(f"Mean:   {s['mean_ms']:.3f} ms")
        print(f"Median: {s['p50_ms']:.3f} ms")
        print(f"P95:    {s['p95_ms']:.3f} ms")
        print(f"Std:    {s['std_ms']:.3f} ms")

    # 4D mask
    else:
        renderer_4d = FGRenderer4D(sess, args.mask)
        # num_views = renderer_4d.v * renderer_4d.u

        # warm up
        if args.render_mode == 'all':
            for _ in range(args.warmup):
                _, _ = renderer_4d.render_all(args.seed)
        elif args.render_mode == 'fg':
            for _ in range(args.warmup):
                _, _ = renderer_4d.render_only_fg(args.seed)

        times = []

        # FG + BG
        if args.render_mode == 'all':
            for i in range(args.iters):
                elapsed, _ = renderer_4d.render_all(args.seed + i)
                times.append(elapsed)
            n_rays = renderer_4d.total_pixels
        # FG
        elif args.render_mode == 'fg':
            for i in range(args.iters):
                elapsed, _ = renderer_4d.render_only_fg(args.seed + i)
                times.append(elapsed)
            n_rays = renderer_4d.fg_total

        s = summarize(times)

        print(f"Resolution: {H}x{W} (batch={BATCH})  Device: {args.device}")
        print(f"Rays:   {n_rays:,}")
        print(f"Mean:   {s['mean_ms']:.3f} ms")
        print(f"Median: {s['p50_ms']:.3f} ms")
        print(f"P95:    {s['p95_ms']:.3f} ms")
        print(f"Std:    {s['std_ms']:.3f} ms")

if __name__ == "__main__":
    main()