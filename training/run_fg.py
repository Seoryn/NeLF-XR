import os, time, random, argparse
from tqdm import tqdm, trange
import imageio
import numpy as np
import cv2
import random
import torch
import torch.nn.functional as F
import glob

from lib import utils, dlfgo_model
from lib.load_data import load_data


def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help='Random seed')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true', 
                        help='do not reload optimizer state from saved ckpt')

    # testing options
    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video_factor", type=float, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # lightfield options
    parser.add_argument("--ray_type", type=str, default='twoplane', help='lightfield type')
    parser.add_argument("--ndc", action='store_true')
    parser.add_argument("--grid_num", type=int, nargs='+', default=0)
    parser.add_argument("--grid_dim", type=int, default=16)
    parser.add_argument("--mlp_depth", type=int, default=4)
    parser.add_argument("--mlp_width", type=int, default=128)
    parser.add_argument("--lr_grid", type=float, default=1e-01)
    parser.add_argument("--wd_grid", type=float, default=0)
    parser.add_argument("--lr_mlp", type=float, default=1e-03)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--load2gpu_on_the_fly", action='store_true')
    parser.add_argument("--basedir", type=str, default='/home/seoryn/onnx/bunny_small_fg_32_32')
    parser.add_argument("--expname", type=str, default='')
    # parser.add_argument("--datadir", type=str, default='/data/igjeong/dataset/stanford_half/bracelet')
    parser.add_argument("--datadir", type=str, default='/data/seoryn/data/bunny')
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--dataset_type", type=str, default='llff')
    parser.add_argument("--dataset_name", type=str, default='stanford')
    parser.add_argument("--epoch", default=1000, type=int)
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--pe", type=int, default=0)
    parser.add_argument("--decomp", type=str, default='4d')
    parser.add_argument("--levels", type=int, default=1)

    # logging/saving options
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save_epoch", type=int, default=250)

    # training only foreground
    parser.add_argument("--train_fg_only", action="store_true")
    parser.add_argument("--val_fg_only", action="store_true")
    parser.add_argument("--bg_lum_thresh", type=float, default=0.30)
    parser.add_argument("--bg_rgb_thresh", type=float, default=0.30)
    parser.add_argument("--save_rgba", action="store_true")

    parser.add_argument("--build_fg_table", action="store_true")
    parser.add_argument("--use_fg_table_at_train", action="store_true")
    parser.add_argument("--use_fg_table_at_render", action="store_true")

    # parser.add_argument("--maskdir", type=str, default="/home/seoryn/onnx/bunny_mask_test", help="(optional) path to precomputed binary masks (png).")
    # parser.add_argument("--mask_thr", type=int, default=128, help="threshold for mask binarization")
    return parser

# FG mask
# torch (for train/val)
def make_fg_mask_rgb(rgb: torch.Tensor, lum_thr=0.02, rgb_thr=0.02):
    # 검정 배경
    r, g, b = rgb.unbind(-1)
    lum = 0.2126*r + 0.7152*g + 0.0722*b
    fg = (lum <= lum_thr) & (torch.max(rgb, dim=-1).values <= rgb_thr)
    return (~fg).float()

    # 흰 배경
    # bg_color = torch.tensor([1.0, 1.0, 1.0], device=rgb.device, dtype=rgb.dtype)
    # dist = torch.norm(rgb - bg_color[None, :], dim=-1)
    # fg = dist > lum_thr
    # return fg.float()

# numpy (for render)
def make_fg_mask_np(rgb_np: np.ndarray, lum_thr=0.02, rgb_thr=0.02):
    # 검정 배경
    # r, g, b = rgb_np[...,0], rgb_np[...,1], rgb_np[...,2]
    # lum = 0.2126*r + 0.7152*g + 0.0722*b
    # fg = (lum <= lum_thr) & (np.max(rgb_np, axis=-1) <= rgb_thr)
    # return ~fg 

    # 흰 배경
    H, W, _ = rgb_np.shape
    b = 10

    border_pixels = np.concatenate([
        rgb_np[:b, :, :].reshape(-1, 3),
        rgb_np[-b:, :, :].reshape(-1, 3),
        rgb_np[:, :b, :].reshape(-1, 3),
        rgb_np[:, -b:, :].reshape(-1, 3),
    ], axis=0)                              # 테두리 픽셀로 배경색 (중앙값) 추정
    bg = np.median(border_pixels, axis=0)   # [3]
    dist = np.linalg.norm(rgb_np - bg[None, None, :], axis=-1)  # 배경색과의 거리(색 차이)
    fg_mask = dist > lum_thr                # dist가 작으면 배경, 크면 전경
    return fg_mask

def get_grid_bounds(model):
    ''' model.grid 내부에서 ray_min, ray_max 반환
    '''
    for m in model.grid.modules():
        if hasattr(m, "ray_min") and hasattr(m, "ray_max"):
            return m.ray_min, m.ray_max
    raise AttributeError("No submodule in model.grid has ray_min/ray_max")

def quantize_stuv_to_uvst_bins(stuv: torch.Tensor, ray_min, ray_max, grid_num):
    ''' 각 ray의 4D 좌표를 grid index로 변환
    '''
    device = stuv.device
    dtype  = stuv.dtype

    # ray_min/max 를 Tensor([4])로
    if not isinstance(ray_min, torch.Tensor):
        ray_min = torch.as_tensor(ray_min, dtype=dtype, device=device)
    else:
        ray_min = ray_min.to(device=device, dtype=dtype)

    if not isinstance(ray_max, torch.Tensor):
        ray_max = torch.as_tensor(ray_max, dtype=dtype, device=device)
    else:
        ray_max = ray_max.to(device=device, dtype=dtype)

    s0, t0, u0, v0 = ray_min.unbind(-1)
    s1, t1, u1, v1 = ray_max.unbind(-1)

    # grid_num을 파이썬 int로
    def _to_int(x):
        if isinstance(x, torch.Tensor):
            return int(x.item())
        return int(x)
    Nu, Nv, Ns, Nt = [_to_int(x) for x in grid_num]  # (u,v,s,t)

    def _bin(x, lo, hi, N):
        t = (x - lo) / (hi - lo + 1e-12)
        b = torch.floor(t * N).long()
        return torch.clamp(b, 0, N - 1)

    s = stuv[:, 0]; t = stuv[:, 1]; u = stuv[:, 2]; v = stuv[:, 3]
    bu = _bin(u, u0, u1, Nu)
    bv = _bin(v, v0, v1, Nv)
    bs = _bin(s, s0, s1, Ns)
    bt = _bin(t, t0, t1, Nt)

    return bu, bv, bs, bt

@torch.no_grad()
def build_fg_uvst_table(args, model, HW, Ks, poses, images, i_train, device):
    Nu, Nv, Ns, Nt = args.grid_num
    occ = torch.zeros((Nu,Nv,Ns,Nt), dtype=torch.bool, device=device)   # UVST table
    ray_min, ray_max = get_grid_bounds(model)
    # use_ext_mask = bool(args.maskdir)   # 외부 마스크 사용 여부
    
    found = 0

    for idx in i_train:
        H, W = HW[idx]
        K = Ks[idx]
        c2w = poses[idx]

        rays = utils.get_rays_of_a_view_twoplane(H, W, K, c2w).to(device)   # [H*W, 4] = [s,t,u,v]
        stuv = rays[:, :4]

        # if use_ext_mask:
        #     base = None
        #     if image_paths is not None and len(image_paths) > idx:
        #         base = os.path.splitext(os.path.basename(image_paths[idx]))[0]
        #     cand = []
        #     if base is not None:
        #         cand.append(os.path.join(args.maskdir, base + ".png"))
        #     cand.append(os.path.join(args.maskdir, f"{idx:03d}.png"))

        #     mask_path = next((p for p in cand if os.path.isfile(p)), None)
        #     if mask_path is not None and os.path.isfile(mask_path):
        #         m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #         if m is None or m.shape[:2] != (H, W):
        #             raise RuntimeError(f"Mask size mismatch: {mask_path} ({None if m is None else m.shape}) vs {(H, W)}")
        #         fg_mask = (m > args.mask_thr)
        #         found += 1
        #     else:
        #         gt = images[idx].cpu().numpy()  # GT RGB
        #         fg_mask = make_fg_mask_np(gt, lum_thr=args.bg_lum_thresh, rgb_thr=args.bg_rgb_thresh)
        # else:
        #     gt = images[idx].cpu().numpy()  # GT RGB
        #     fg_mask = make_fg_mask_np(gt, lum_thr=args.bg_lum_thresh, rgb_thr=args.bg_rgb_thresh)
        
        # GT RGB 기반으로 FG mask 계산
        gt = images[idx].cpu().numpy()  # [H,W,3], 0~1
        fg_mask = make_fg_mask_np(gt, lum_thr=args.bg_lum_thresh, rgb_thr=args.bg_rgb_thresh)

        if not fg_mask.any():
            continue
        
        # fg_mask = True인 픽셀의 index
        fg_idx = torch.from_numpy(np.flatnonzero(fg_mask.reshape(-1))).to(device)
        stuv_fg = stuv[fg_idx]
        bu, bv, bs, bt = quantize_stuv_to_uvst_bins(stuv_fg, ray_min, ray_max, args.grid_num)
        occ[bu, bv, bs, bt]=True

        # dilation
        for ds in [-1, 0, 1]:
            for dt in [-1, 0, 1]:
                bs_dilated = torch.clamp(bs + ds, 0, Ns - 1)
                bt_dilated = torch.clamp(bt + dt, 0, Nt - 1)
                occ[bu, bv, bs_dilated, bt_dilated] = True

    print(f"[FG] external masks used: {found}/{len(i_train)}")
    return occ

def load_fg_table(args, device):
    ''' return occ, grid_num, ray_min, ray_max if fg_table_path
    '''
    fg_table_path = os.path.join(args.basedir,args.expname,'fg_uvst_table.pt')
    if os.path.isfile(fg_table_path):
        blob = torch.load(fg_table_path,map_location='cpu')
        return (blob['occ'].to(device).bool(),
                torch.tensor(blob['grid_num'],device=device),
                blob['ray_min'].to(device),
                blob['ray_max'].to(device))
    return None,None,None,None

@torch.no_grad()
def warmup(model, ray_type, H, W, K, c2w):
    ''' Warm up the model for faster inference.
    '''
    if ray_type == 'twoplane':
        ray = utils.get_rays_of_a_view_twoplane(H, W, K, c2w)
    else:
        raise ValueError(f'Unknown ray_type: {ray_type}')
    ray = ray.to(next(model.parameters()).device)
    _ = model(ray).reshape(H, W, -1)
    del _

@torch.no_grad()
def render_viewpoints(model, ray_type, render_set, render_poses, HW, Ks,
                      gt_imgs=None, savedir=None, dump_images=True, expdir=None, render_factor=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,
                      bg_lum_thresh=0.02, bg_rgb_thresh=0.02, save_rgba=False,
                      occ=None, ray_min_t=None, ray_max_t=None, grid_num=None):
    ''' Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    if render_factor != 0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW / render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    render_time = []
    H, W = HW[0]
    K = Ks[0]

    # warm up
    warmup(model, ray_type, H, W, K, render_poses[0])

    if ray_type == 'twoplane':
        for i, c2w in enumerate(tqdm(render_poses)):
            c2w = torch.Tensor(c2w)
            ray = utils.get_rays_of_a_view_twoplane(H, W, K, c2w)
            ray = ray.to(next(model.parameters()).device)

            fg_mask_img = False
            used_fg = False  # 이번 프레임에서 FG 테이블을 실제 사용했는지 표시

            # 타이머
            t0 = torch.cuda.Event(True)
            t1 = torch.cuda.Event(True)
            import time
            wall0 = time.perf_counter()

            if (occ is not None) and (grid_num is not None) and (ray_min_t is not None) and (ray_max_t is not None):
                stuv = ray[:, :4]  # [s,t,u,v]
                bu, bv, bs, bt = quantize_stuv_to_uvst_bins(stuv, ray_min_t, ray_max_t, grid_num)   # ray → uvst bin idx
                keep = occ[bu, bv, bs, bt]  # model ray

                if keep.any():
                    Nu, Nv, Ns, Nt = [int(x) for x in grid_num]
                    bin_lin = (((bu * Nv) + bv) * Ns + bs) * Nt + bt    # 4D idx → 1D idx
                    idx_fg = torch.nonzero(keep, as_tuple=False).squeeze(1) # keep=True인 픽셀의 idx
                    bin_lin_fg = bin_lin[idx_fg]

                    # 같은 bin끼리 이웃하도록 정렬
                    order = torch.argsort(bin_lin_fg)
                    idx_sorted = idx_fg[order]  # [K]
                    
                    # H*W개의 픽셀에 대한 RGB 정보를 담을 버퍼
                    pred_dev = torch.zeros((H*W, 3), device=ray.device, dtype=ray.dtype)

                    # 타이머 시작
                    t0.record()
                    
                    CHUNK = 262144
                    ptr = 0
                    n = idx_sorted.numel()  # 전경 픽셀 수

                    # CHUNK 단위로
                    while ptr < n:
                        end = min(ptr + CHUNK, n)
                        batch_idx = idx_sorted[ptr:end]
                        pred_dev[batch_idx] = model(ray[batch_idx]) # 전경에 해당하는 ray만 forward
                        ptr = end

                    pred = pred_dev.view(H, W, 3).detach().cpu().numpy()    # (H, W, 3)으로 이미지 구성
                    fg_mask_img = keep.view(H, W).detach().cpu().numpy()    # keep을 (H, W)로 구성
                    used_fg = True

            # FG 경로를 쓰지 못했다면 전체 프레임 추론
            if not used_fg:
                t0.record()
                pred = model(ray).reshape(H, W, -1).detach().cpu().numpy()

            # 타이머 종료
            t1.record(); torch.cuda.synchronize()
            gpu_ms = torch.cuda.Event.elapsed_time(t0, t1)
            wall_ms = (time.perf_counter() - wall0) * 1000.0
            render_time.append(gpu_ms)
            rgbs.append(pred)
            # if i == 0:
            #     print('Testing', pred.shape)

            # 지표 계산
            if gt_imgs is not None and render_factor == 0 and i < len(gt_imgs):
                p = -10. * np.log10(np.mean(np.square(pred - gt_imgs[i])))
                psnrs.append(p)
                if eval_ssim:
                    ssims.append(utils.rgb_ssim(pred, gt_imgs[i], max_val=1))
                if eval_lpips_alex:
                    lpips_alex.append(utils.rgb_lpips(pred, gt_imgs[i], net_name='alex', device=c2w.device))
                if eval_lpips_vgg:
                    lpips_vgg.append(utils.rgb_lpips(pred, gt_imgs[i], net_name='vgg', device=c2w.device))

            # mask_2d_from_file = None
            # if mask_paths is not None:
            #     mp = mask_paths[i]
            #     if os.path.isfile(mp):
            #         m = imageio.imread(mp)
            #         if m.ndim == 3: m = m[..., 0]
            #         if m.shape != (H, W):
            #             m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            #         mask_2d_from_file = (m > mask_thr)

            # alpha_2d = None
            # if mask_2d_from_file is not None:
            #     alpha_2d = mask_2d_from_file
            # elif fg_mask_img is not None:
            #     alpha_2d = fg_mask_img

            alpha_2d = fg_mask_img

            if alpha_2d is not None:
                pred = pred * alpha_2d[..., None]

            # 저장
            if savedir is not None and dump_images:
                os.makedirs(savedir, exist_ok=True)
                if save_rgba and (alpha_2d is not None):
                    rgba8 = np.concatenate([utils.to8b(pred), (alpha_2d.astype(np.uint8)*255)[..., None]], axis=-1)
                    imageio.imwrite(os.path.join(savedir, f'{i:03d}.png'), rgba8)
                else:
                    imageio.imwrite(os.path.join(savedir, f'{i:03d}.png'), utils.to8b(pred))

    if len(psnrs):
        test_str = f'Set: {render_set} / Testing psnr: {np.mean(psnrs)} (avg) '
        if eval_ssim: test_str += f'/ ssim: {np.mean(ssims)} '
        if eval_lpips_vgg: test_str += f'/ lpips(vgg): {np.mean(lpips_vgg)} '
        if eval_lpips_alex: test_str += f'/ lpips(alex): {np.mean(lpips_alex)} '
        if len(render_time): test_str += f'/ render time: {np.mean(render_time)} ms '
        print(test_str)
        if expdir is not None:
            with open(f'{expdir}/test_results.txt', 'a') as f:
                f.write(test_str + '\n')

    return np.array(rgbs)

def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def load_everything(args):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(args)
    kept = {'hwf', 'HW', 'Ks', 'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images', 'focal_depth', 'image_paths'}
    # remove useless field
    for k in list(data_dict.keys()):
        if k not in kept:
            data_dict.pop(k)
    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])

    if 'image_paths' not in data_dict or not data_dict['image_paths']:
        imgdir = os.path.join(args.datadir, 'images')
        paths = []
        for ext in ('*.png', '*jpg', '*jpeg'):
            paths += sorted(glob.glob(os.path.join(imgdir, ext)))
        data_dict['image_paths'] = paths

    return data_dict

def print_cuda_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"Allocated: {allocated:.2f} MB / Reserved: {reserved:.2f} MB")

def create_new_model(args, ray_min, ray_max, device):
    print(f'\033[96muse LightField {args.ray_type} grid\033[0m')
    if args.ray_type == 'twoplane':
        model = dlfgo_model.DLFGO_twoplane(
            ray_min=ray_min, ray_max=ray_max, grid_num=args.grid_num,
            ray_type='twoplane', grid_dim=args.grid_dim,
            mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
            pe=args.pe, decomp=args.decomp, levels=args.levels)
    else:
        raise ValueError(f'Unknown ray_type: {args.ray_type}')
    model = model.to(device)
    optimizer = torch.optim.Adam([
        {'params': model.grid.parameters(), 'lr': args.lr_grid},
        {'params': model.mlp.parameters(), 'lr': args.lr_mlp}
    ])
    return model, optimizer

def load_existed_model(args, reload_ckpt_path, device):
    if args.ray_type == 'twoplane':
        model_class = dlfgo_model.DLFGO_twoplane
    else:
        raise ValueError(f'Unknown ray_type: {args.ray_type}')
    model = utils.load_model(model_class, reload_ckpt_path).to(device)
    optimizer = torch.optim.Adam([
        {'params': model.grid.parameters(), 'lr': args.lr_grid},
        {'params': model.mlp.parameters(), 'lr': args.lr_mlp}
    ])
    model, optimizer, start = utils.load_checkpoint(
        model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start

def scene_rep_reconstruction(args, data_dict, device):
    HW, Ks, i_train, i_val, i_test, poses, render_poses, images, focal_depth = [
        data_dict[k] for k in ['HW', 'Ks', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'focal_depth']]

    last_ckpt_path = os.path.join(args.basedir, args.expname, 'train_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print('scene_rep_reconstruction: train from scratch')
        rgb_all = images[i_train].to('cpu' if args.load2gpu_on_the_fly else device)
        ray_min, ray_max = utils.get_ray_min_max(
            ray_type=args.ray_type, rgb_all=rgb_all,
            train_poses=poses[i_train], HW=HW[i_train], Ks=Ks[i_train])
        model, optimizer = create_new_model(args, ray_min, ray_max, device)
        start = 0
    else:
        print(f'scene_rep_reconstruction: reload from {reload_ckpt_path}')
        model, optimizer, start = load_existed_model(args, reload_ckpt_path, device)

    # FG table 만들고 저장
    fg_table_path = os.path.join(args.basedir, args.expname, 'fg_uvst_table.pt')
    occ = None
    if args.build_fg_table:
        print("[FG] Building UVST foreground table...")
        occ = build_fg_uvst_table(args, model, HW, Ks, poses, images, i_train, device)
        rm, rM = get_grid_bounds(model)
        torch.save({'occ': occ.bool().cpu(), 'grid_num': args.grid_num,
                    'ray_min': rm.detach().cpu(), 'ray_max': rM.detach().cpu()},
                    fg_table_path)
        print(f"[FG] Saved UVST table: {fg_table_path}")

    else:
        if os.path.isfile(fg_table_path):
            blob = torch.load(fg_table_path, map_location='cpu')
            occ = blob['occ'].to(device)
            rm = blob['ray_min']
            rM = blob['ray_max']
            print(f"[FG] Loaded UVST table: {fg_table_path}")

    # rays/gt
    def gather_training_rays(ray_type):
        rgb_train_original = images[i_train].to('cpu' if args.load2gpu_on_the_fly else device)
        rgb_val_original = images[i_val].to('cpu' if args.load2gpu_on_the_fly else device)
        rgb_val, ray_val = utils.get_training_rays(
            ray_type=ray_type, rgb_original=rgb_val_original,
            train_poses=poses[i_val], HW=HW[i_val], Ks=Ks[i_val])
        rgb_train, ray_train = utils.get_training_rays(
            ray_type=ray_type, rgb_original=rgb_train_original,
            train_poses=poses[i_train], HW=HW[i_train], Ks=Ks[i_train])
        
        index_generator = utils.batch_indices_generator(len(rgb_train), args.batch_size)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_train, ray_train, rgb_val, ray_val, batch_index_sampler
    
    # init batch rays sampler
    rgb_train, ray_train, rgb_val, ray_val, batch_index_sampler = gather_training_rays(args.ray_type)

    # FG filter by UVST table
    if occ is not None and args.use_fg_table_at_train:
        with torch.no_grad():
            stuv = ray_train[:, :4]   # [s,t,u,v]
            rm, rM = get_grid_bounds(model)
            bu, bv, bs, bt = quantize_stuv_to_uvst_bins(stuv, rm, rM, args.grid_num)  # [N,4]
            keep = occ[bu, bv, bs, bt]  # [N] bool
            if keep.any():
                rgb_train = rgb_train[keep]
                ray_train = ray_train[keep]
                print(f"[FG/UVST] Train filtered: {int(keep.sum())} / {len(keep)} rays kept")
            else:
                print("[WARN] UVST table kept 0 rays; fallback to all rays.")

        # 배치 인덱스 갱신
        index_generator = utils.batch_indices_generator(len(rgb_train), args.batch_size)
        batch_index_sampler = lambda: next(index_generator)
                
    # 배경 제거 모델만 학습
    if args.train_fg_only:
        with torch.no_grad():
            m_train = make_fg_mask_rgb(rgb_train, lum_thr=args.bg_lum_thresh, rgb_thr=args.bg_rgb_thresh) # 모델 마스크 계산
            idx_fg = torch.nonzero(m_train > 0.5, as_tuple=False).squeeze(1) # 모델 픽셀의 idx
        
        if idx_fg.numel() == 0:
            print("[WARN] train_fg_only ON, but no foreground rays. Proceeding with all rays.")
        else:
            # 학습 데이터를 모델만 남김
            rgb_train = rgb_train[idx_fg]
            ray_train = ray_train[idx_fg]

            # 모델 픽셀 수에 맞게 배치 사이즈 조절
            if len(rgb_train) < args.batch_size:
                print(f"[INFO] FG rays < batch_size: adjust batch_size to {len(rgb_train)}")
                args.batch_size = len(rgb_train)

            index_generator = utils.batch_indices_generator(len(rgb_train), args.batch_size)
            batch_index_sampler = lambda: next(index_generator)
            fg_frac = float(idx_fg.numel()) / float(m_train.numel())
            print(f"[INFO] Train FG-only: {idx_fg.numel()} / {m_train.numel()} rays ({fg_frac*100:.2f}%)")

    # 배경 제거했을 때의 PSNR 측정
    if args.val_fg_only and len(rgb_val) > 0:
        with torch.no_grad():
            m_val = make_fg_mask_rgb(rgb_val, lum_thr=args.bg_lum_thresh, rgb_thr=args.bg_rgb_thresh)
            idx_fg_v = torch.nonzero(m_val > 0.5, as_tuple=False).squeeze(1)
        if idx_fg_v.numel() == 0:
            print("[WARN] val_fg_only ON, but no foreground rays for val.")
        else:
            rgb_val = rgb_val[idx_fg_v]
            ray_val = ray_val[idx_fg_v]
            fg_frac_val = float(idx_fg_v.numel()) / float(m_val.numel())
            print(f"[INFO] Val FG-only: {idx_fg_v.numel()} / {m_val.numel()} rays ({fg_frac_val*100:.2f}%)")

    epoch = args.epoch
    iter_per_epoch = max(1, len(rgb_train) // args.batch_size)
    iters = epoch * iter_per_epoch
    print(f'Training rays: {len(rgb_train)} / Batch: {args.batch_size} / Epoch: {epoch} / Iter per epoch: {iter_per_epoch}')

    torch.cuda.empty_cache()
    psnr_lst, psnr_val_lst = [], []
    time0 = time.time()
    global_step = -1
    print("before training, cuda memory usage check")
    print_cuda_memory_usage()

    pbar = trange(1 + start, 1 + iters)
    for global_step in pbar:
        selected_indices = batch_index_sampler()
        rgb_train_batch = rgb_train[selected_indices]
        ray_train_batch = ray_train[selected_indices]

        optimizer.zero_grad(set_to_none=True)

        render_result = model(ray_train_batch, global_step)
        loss = F.mse_loss(render_result, rgb_train_batch)
        psnr = utils.mse2psnr(loss.detach())
        (loss * 1000).backward()
        optimizer.step()
        psnr_lst.append(psnr.item())
        pbar.set_description(f"Loss: {float(loss):.7f}, PSNR: {psnr.item():.2f}")

        # update lr
        decay_factor = 0.995 ** (1 / iter_per_epoch)
        for g in optimizer.param_groups:
            g['lr'] *= decay_factor

        # check log & save
        if global_step % (1 * iter_per_epoch) == 0:
            eps = time.time() - time0
            s = f'expname / {args.expname} / FGonly(Train/Val) / {int(args.train_fg_only)}/{int(args.val_fg_only)} / '
            s += f'Eps / {eps//3600:02.0f}:{eps//60%60:02.0f}:{eps%60:02.0f} / '
            s += f'grid lr / {optimizer.param_groups[0]["lr"]:.8f} / mlp lr / {optimizer.param_groups[1]["lr"]:.8f} / '
            s += f'Loss / {float(loss):.7f} / step / {global_step:6d} / epoch / {global_step//iter_per_epoch:4d} / '
            s += f'Train PSNR / {np.mean(psnr_lst):5.2f} / '
            with torch.no_grad():
                if len(rgb_val) > 0:
                    pred_v = model(ray_val, global_step)
                    mse_v = F.mse_loss(pred_v, rgb_val)
                    psnr_v = utils.mse2psnr(mse_v.detach()).item()
                    psnr_val_lst.append(psnr_v)
                    s += f'Val PSNR / {np.mean(psnr_val_lst):5.2f} / '
            tqdm.write(s)
            os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
            with open(os.path.join(args.basedir, args.expname, 'training_result.txt'), 'a') as f:
                f.write(s + '\n')
            psnr_lst, psnr_val_lst = [], []

        if global_step % (args.save_epoch * iter_per_epoch) == 0:
            model_path = os.path.join(args.basedir, args.expname, 'model',
                                      f'train_last_{global_step//(iter_per_epoch)}epoch.tar')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
            print('saved:', model_path)

    if global_step != -1:
        last = os.path.join(args.basedir, args.expname, 'train_last.tar')
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last)
        print('saved last:', last)
        weight_path = os.path.join(args.basedir, args.expname, 'train_last_weight.pt')
        torch.save({'model_kwargs': model.get_kwargs(), 
                    'model_state_dict': model.state_dict()}, weight_path)
        print('saved weights:', weight_path)

def train(args, data_dict, device):
    torch.cuda.empty_cache()
    t0 = time.time()
    os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
    os.makedirs(os.path.join(args.basedir, args.expname, 'model'), exist_ok=True)
    with open(os.path.join(args.basedir, args.expname, 'args.txt'), 'w') as f:
        for arg in sorted(vars(args)):
            f.write(f'{arg} = {getattr(args, arg)}\n')

    scene_rep_reconstruction(args, data_dict, device)
    t = time.time() - t0
    print('train: finish (eps time %02.0f:%02.0f:%02.0f)' % (t//3600, t//60%60, t%60))

def collect_mask_paths(indices, data_dict, maskdir, thr):
    if not maskdir or 'image_paths' not in data_dict:
        return None
    paths = []
    for i in indices:
        base = os.path.splitext(os.path.basename(data_dict['image_paths'][i]))[0] + '.png'
        paths.append(os.path.join(maskdir, base))
    return paths

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    data_name = os.path.basename(args.datadir)
    if args.grid_num == 0:
        if data_name == "beans":
            args.grid_num = [8, 8, 128, 64]
        elif data_name == "bracelet":
            args.grid_num = [8, 8, 128, 80]
        elif data_name == "gem":
            args.grid_num = [8, 8, 96, 128]
        elif data_name == "truck":
            args.grid_num = [8, 8, 160, 120]
        elif data_name == "chess":
            args.grid_num = [8, 8, 175, 100]
        elif data_name == "bulldozer":
            args.grid_num = [8, 8, 192, 144]
        elif data_name == "flowers":
            args.grid_num = [8, 8, 160, 192]
        elif data_name == "treasure":
            args.grid_num = [8, 8, 192, 160]
        else:
            args.grid_num = [8, 8, 128, 128]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(args.gpuid)
        print('>>> Using GPU:', args.gpuid)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print('>>> Using CPU')

    seed_everything()
    data_dict = load_everything(args)

    if not args.render_only:
        train(args, data_dict, device)

    # load model for rendering
    if args.render_test or args.render_train:
        ckpt_path = os.path.join(args.basedir, args.expname, 'train_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if args.ray_type == 'twoplane':
            model_class = dlfgo_model.DLFGO_twoplane
        else:
            raise ValueError(f'Unknown ray_type: {args.ray_type}')
        model = utils.load_model(model_class, ckpt_path).to(device)

    # load UVST table for rendering
    occ = ray_min_t = ray_max_t = grid_num_t = None
    if args.use_fg_table_at_render:
            occ, grid_num_t, ray_min_t, ray_max_t = load_fg_table(args, device)
            if occ is None:
                print("[FG] No UVST table found for render; will render all rays.")

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(args.basedir, args.expname, f'render_train_{ckpt_name}')
        expdir = os.path.join(args.basedir, args.expname)
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)

        train_mask_paths = collect_mask_paths(data_dict['i_train'], data_dict, args.maskdir, args.mask_thr)

        rgbs = render_viewpoints(
            model, args.ray_type, 'train',
            data_dict['poses'][data_dict['i_train']],
            data_dict['HW'][data_dict['i_train']],
            data_dict['Ks'][data_dict['i_train']],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
            savedir=testsavedir, dump_images=args.dump_images, expdir=expdir,
            eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
            bg_lum_thresh=args.bg_lum_thresh, bg_rgb_thresh=args.bg_rgb_thresh,
            save_rgba=args.save_rgba,                     
            render_factor=args.render_video_factor,
            occ=occ, ray_min_t=ray_min_t, ray_max_t=ray_max_t,
            grid_num=(grid_num_t if grid_num_t is not None else args.grid_num)
        )
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(args.basedir, args.expname, f'render_test_{ckpt_name}')
        expdir = os.path.join(args.basedir, args.expname)
        os.makedirs(testsavedir, exist_ok=True)

        test_mask_paths = collect_mask_paths(data_dict['i_test'], data_dict, args.maskdir, args.mask_thr)
        rgbs = render_viewpoints(
            model, args.ray_type, 'test',
            data_dict['poses'][data_dict['i_test']],
            data_dict['HW'][data_dict['i_test']],
            data_dict['Ks'][data_dict['i_test']],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
            savedir=testsavedir, dump_images=args.dump_images, expdir=expdir,
            eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
            bg_lum_thresh=args.bg_lum_thresh, bg_rgb_thresh=args.bg_rgb_thresh,
            save_rgba=args.save_rgba,
            render_factor=args.render_video_factor,
            occ=occ, ray_min_t=ray_min_t, ray_max_t=ray_max_t,
            grid_num=(grid_num_t if grid_num_t is not None else args.grid_num)
        )
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)

    print('Done')
