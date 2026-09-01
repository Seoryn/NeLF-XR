import torch
import numpy as np
import imageio
import time
import onnxruntime as ort
from lib.load_data import load_data
from lib import utils
import argparse

TARGET_H = 512
TARGET_W = 512

def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ray_type", type=str, default='twoplane', help='lightfield type')
    parser.add_argument("--ndc", action='store_true')
    parser.add_argument("--basedir", type=str, default='/home/seoryn/onnx/bunny_fg')
    parser.add_argument("--expname", type=str, default='')
    parser.add_argument("--datadir", type=str, default='/data/seoryn/data/bunny')
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--dataset_type", type=str, default='llff')
    parser.add_argument("--dataset_name", type=str, default='stanford')
    parser.add_argument("--decomp", type=str, default='4d')
    
    parser.add_argument("--model", default='/home/seoryn/onnx/dlfgo_onnx/models_fg/bunny_small_fg_32_32.onnx')
    parser.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    return parser

def quantize_stuv_to_uvst_bins_np(stuv, ray_min, ray_max, grid_num):
    ''' 학습 때와 동일하게 4D UVST quantization
    '''
    s0, t0, u0, v0 = ray_min
    s1, t1, u1, v1 = ray_max
    Nu, Nv, Ns, Nt = grid_num

    def to_idx(val, v_min, v_max, Nbin):
        idx = (val - v_min) / (v_max - v_min + 1e-8) * Nbin
        idx = np.floor(idx).astype(np.int32)
        return np.clip(idx, 0, Nbin - 1)

    s = stuv[:,0]; t = stuv[:,1]
    u = stuv[:,2]; v = stuv[:,3]

    bu = to_idx(u, u0, u1, Nu)
    bv = to_idx(v, v0, v1, Nv)
    bs = to_idx(s, s0, s1, Ns)
    bt = to_idx(t, t0, t1, Nt)
    return bu, bv, bs, bt

def get_dataset_rays_for_one_view(args, device, target_hw=(TARGET_H, TARGET_W)):
    ''' 학습 때 사용된 카메라에서 ray 추출
    '''
    data_dict = load_data(args)
    HW      = data_dict['HW']        # [N, 2]
    Ks      = data_dict['Ks']        # [N, 3, 3]
    poses   = data_dict['poses']     # [N, 3, 4]
    i_train = data_dict['i_train']

    # 학습에 사용된 첫 번째 뷰 사용
    idx = i_train[0].item() if torch.is_tensor(i_train[0]) else i_train[0]

    H, W = HW[idx]          # 원래 데이터 해상도
    K    = Ks[idx].copy()
    c2w  = poses[idx]

    Ht, Wt = target_hw      # 원하는 해상도

    sx = Wt / float(W)
    sy = Ht / float(H)
    K[0, 0] *= sx       # fx
    K[0, 2] *= sx       # cx
    K[1, 1] *= sy       # fy
    K[1, 2] *= sy       # cy

    ray = utils.get_rays_of_a_view_twoplane(Ht, Wt, K, c2w)

    # DEBUG
    ray_np = ray.cpu().numpy()
    sample_pixels = [
        0,                          # 좌상단
        W - 1,                      # 우상단
        (H // 2) * W + (W // 2),    # 중앙
        (H - 1) * W,                # 좌하단
        (H - 1) * W + (W - 1)       # 우하단
    ]

    print("=== PYTHON RAYS ===")
    for p in sample_pixels:
        s, t, u, v = ray_np[p]
        print(f"pixel={p} (x={p % W}, y={p // W}) -> u={u}, v={v}, s={s}, t={t}")
    print("===================")

    return ray, H, W

# ONNX Runtime
def build_session(onnx_path, device="cuda", intra_threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if device == "cuda":
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,                    # GPU 0번
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "DEFAULT",
                }
            ),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    print("[ORT] Active Providers:", sess.get_providers())   # GPU 사용 확인
    return sess

def render_fg_image_onnx(args, onnx_path, fg_table_path, out_path):
    # 1) ONNX 세션 로드
    # sess = ort.InferenceSession(
    #     onnx_path,
    #     providers=[("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
    # )
    sess = build_session(args.model, args.device, args.threads)

    # 2) FG mask 로드
    blob = torch.load(fg_table_path, map_location="cpu")
    occ_np   = blob["occ"].numpy().astype(bool)   # [Nu,Nv,Ns,Nt] (u,v,s,t)
    grid_num = blob["grid_num"]
    ray_min  = blob["ray_min"].numpy()            # [s0,t0,u0,v0]
    ray_max  = blob["ray_max"].numpy()            # [s1,t1,u1,v1]

    Nu, Nv, Ns, Nt = grid_num

    print(">>> FG table:")
    print("  occ shape:", occ_np.shape)
    print("  occ True count:", occ_np.sum())
    print("  grid_num:", grid_num)
    print("  ray_min:", ray_min)
    print("  ray_max:", ray_max)

    # 3) 데이터셋 기반 카메라에서 ray 생성
    ray_torch, H, W = get_dataset_rays_for_one_view(args, device="cpu")
    rays_np = ray_torch.numpy()  # [H*W,4]
    N = rays_np.shape[0]

    # 4) UVST quantization → FG pixel
    stuv = rays_np[:, :4]
    bu, bv, bs, bt = quantize_stuv_to_uvst_bins_np(stuv, ray_min, ray_max, grid_num)

    fg_mask_flat = occ_np[bu, bv, bs, bt]
    fg_idx = np.nonzero(fg_mask_flat)[0]
    print(f"[FG] {fg_idx.size} / {N} rays ({100*fg_idx.size/N:.1f}%)")

    warmup = 5
    iters = 30

    # ALL Rendering Time 측정
    rays_fp32 = np.ascontiguousarray(rays_np.astype(np.float32))
    print("\n[ALL] Full rays:", rays_fp32.shape[0])

    for _ in range(warmup):
        _ = sess.run(None, {"ray": rays_fp32})

    full_times = []
    for i in range(iters):
        t0 = time.perf_counter()
        _ = sess.run(None, {"ray": rays_fp32})
        t1 = time.perf_counter()
        full_times.append(t1 - t0)
    
    full_times = np.array(full_times) * 1000
    print(f"[ALL] Full rendering time (mean): {full_times.mean():.3f} ms")

    # FG Rendering Time 측정
    if fg_idx.size > 0:
        fg_rays = np.ascontiguousarray(rays_fp32[fg_idx])
        print(f"[FG] FG-only rays: {fg_rays.shape[0]} ({(fg_rays.shape[0] / rays_fp32.shape[0])*100:.1f}%)")
    
    for _ in range(warmup):
        _ = sess.run(None, {"ray": fg_rays})

    fg_times = []
    for i in range(iters):
        t0 = time.perf_counter()
        _ = sess.run(None, {"ray": fg_rays})
        t1 = time.perf_counter()
        fg_times.append(t1 - t0)
         
    fg_times = np.array(fg_times) * 1000
    print(f"[FG] FG-only rendering time (mean): {fg_times.mean():.3f} ms")

    # 5) FG ray만 ONNX 모델에 입력
    rgb_flat = np.zeros((N, 3), dtype=np.float32)
    # rgb_flat[:] = [190/255, 203/255, 201/255]           # bracelet
    if fg_idx.size > 0:
        rays_fg = rays_np[fg_idx].astype(np.float32)
        rgb_fg = sess.run(None, {"ray": rays_fg})[0]
        rgb_flat[fg_idx] = rgb_fg

    # 6) 저장
    img = rgb_flat.reshape(TARGET_H, TARGET_W, 3)
    img = np.clip(img, 0.0, 1.0)
    imageio.imwrite(out_path, (img * 255).astype(np.uint8))
    print(f"\n[IMG] RGB 저장 : {out_path}")

    alpha = (fg_mask_flat.reshape(TARGET_H, TARGET_W).astype(np.uint8) * 255)
    rgba = np.concatenate([(img * 255).astype(np.uint8), alpha[..., None]], axis=-1)
    imageio.imwrite("output_rgba.png", rgba)
    print(f"[IMG] RGBA 저장: output_rgba.png")

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args([])

    render_fg_image_onnx(
        args=args,
        onnx_path="model.onnx",
        fg_table_path="fg_uvst_table.pt",
        out_path="output.png"
    )
