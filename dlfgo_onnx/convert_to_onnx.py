import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
from lib import utils, dlfgo_model
from lib.dlfgo_model import DLFGO_twoplane
import time
import onnx
import onnxruntime as ort
# import onnxoptimizer

def to8b(x):
    """Convert [0,1] to [0,255]."""
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def config_parser():
    parser = argparse.ArgumentParser(description='Convert dlf2dgo model to ONNX')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output_path', type=str, default='model.onnx', help='Path to save the ONNX model')
    parser.add_argument('--test_image_path', type=str, default='onnx_image.png', help='Path to save the test image')
    parser.add_argument('--gpuid', type=int, default=1, help='GPU ID')
    
    # bg 제거 관련
    parser.add_argument('--occ_mask_path', type=str, required=False, default=None, help='occ_mask.pt 경로')
    parser.add_argument('--occ_tau', type=float, default=0.5)              # 임계값
    parser.add_argument('--occ_reduce', type=str, default='mean', choices=['max','mean'])  # 여러 level에서 나온 mask를 합치는 기준 설정
    parser.add_argument('--bg_val', type=float, nargs='+', default=[0.0])  # 배경 색 설정
    return parser

def load_model(ckpt_path, use_occ_mask=None, occ_tau=0.5, occ_reduce='max', bg_val=0.0):
    print(f"Loading model from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # 원본 모델 로드
    # original_model = utils.load_model(dlfgo_model.DLFGO_twoplane, ckpt_path, strict=False)
    # original_model.eval()
    # 
    # return original_model
    
    saved_kwargs = {}
    if isinstance(ckpt, dict) and 'model_kwargs' in ckpt and isinstance(ckpt['model_kwargs'], dict):
        saved_kwargs = ckpt['model_kwargs']

    # 필수 파라미터
    ray_min   = saved_kwargs.get('ray_min',   ckpt.get('ray_min'))
    ray_max   = saved_kwargs.get('ray_max',   ckpt.get('ray_max'))
    grid_num  = saved_kwargs.get('grid_num',  ckpt.get('grid_num'))
    grid_dim  = saved_kwargs.get('grid_dim',  ckpt.get('grid_dim', 16))
    decomp    = saved_kwargs.get('decomp',    '4d')
    levels    = saved_kwargs.get('levels',    [1])
    mlp_depth = saved_kwargs.get('mlp_depth', 2)
    mlp_width = saved_kwargs.get('mlp_width', 128)
    pe        = saved_kwargs.get('pe',        0)
    interp_mode = int(saved_kwargs.get('interp_mode', 16))

    # use_occ_mask를 체크포인트에서 자동 감지
    if use_occ_mask is None:
        use_occ_mask = saved_kwargs.get('use_occ_mask', False)
        print(f"[INFO] Auto-detected use_occ_mask={use_occ_mask} from checkpoint")
    
    # occ_tau, bg_val도 체크포인트 우선
    if use_occ_mask:
        occ_tau = saved_kwargs.get('occ_tau', occ_tau)
        bg_val = saved_kwargs.get('bg_val', bg_val)
        occ_reduce = saved_kwargs.get('occ_reduce', occ_reduce)
        print(f"[INFO] Occupancy params: tau={occ_tau}, reduce={occ_reduce}, bg_val={bg_val}")

    # bg_val
    if isinstance(bg_val, list) and len(bg_val) == 1:
        bg_val = bg_val[0]

    model = DLFGO_twoplane(
        ray_min=ray_min, ray_max=ray_max, grid_num=grid_num,
        grid_dim=grid_dim, mlp_depth=mlp_depth, mlp_width=mlp_width, pe=pe,
        decomp=decomp, levels=levels, interp_mode=interp_mode,
        use_occ_mask=bool(use_occ_mask), occ_tau=float(occ_tau),
        occ_reduce=occ_reduce, bg_val=bg_val
    ).eval()

    state = None
    if isinstance(ckpt, dict):
        state = ckpt.get('model_state_dict') or ckpt.get('state_dict')
    if state is None:
        state = ckpt
    # state = _pad_occ_channel_if_needed(model, state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[INFO] missing keys:", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("[INFO] unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")
    return model

def load_mask_tensor(mask_path, H, W, device='cpu'):
    '''
    [1, H*W, 1] 형태의 float32 {0,1} 텐서 반환
    '''
    m = imageio.v2.imread(mask_path)   # HxW or HxWxC, 0~255
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 127).astype(np.float32)   # 0/1
    m = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    if m.shape[-2:] != (H, W):
        m = F.interpolate(m, size=(H, W), mode='nearest')
    m = m.view(1, -1, 1)  # [1, H*W, 1]
    return m.to(device)

def get_test_rays(H, W):
    """
    테스트 이미지 생성을 위한 광선 생성 (twoplane 방식)
    """
    # 카메라 파라미터 설정 (빈 텐서)
    K = torch.zeros(3, 3, dtype=torch.float32)

    # 카메라 포즈 설정 (빈 텐서)
    c2w = torch.zeros(3, 4, dtype=torch.float32)
    
    # 카메라 위치 설정 (-1에서 1 사이의 값)
    c2w[0, 3] = 0.0 # x축 중심
    c2w[1, 3] = 0.0 # y축 중심
    # c2w[2, 3] = 0.0
    
    # 광선 생성
    ray = utils.get_rays_of_a_view_twoplane(H, W, K, c2w)
    return ray

def convert_to_onnx(model, output_path, H, W, device, only_fg=False, mask_path=None):
    # 테스트 입력 생성
    ray = get_test_rays(H, W)
    ray = ray.to(device)

    # 입력 형태 확인
    print(f"Input shape: {ray.shape}")
    
    # 모델 추론 테스트
    with torch.no_grad():
        output = model(ray, global_step='debug')
    
    # 출력 형태 확인
    print(f"Output shape: {output.shape}")
    
    # 원본 모델로 이미지 생성
    rgb_numpy = output.reshape(H, W, 3).cpu().numpy()
    imageio.imwrite(f"torch_test.png", to8b(rgb_numpy))
    print(f"PyTorch test image saved to torch_test.png")


    mask = torch.load(mask_path)
    occ_np = mask["occ"].numpy()
    grid_num = mask["grid_num"]
    ray_min = mask["ray_min"].numpy()
    ray_max = mask["ray_max"].numpy()

    generate_fg_image_from_model(model, occ_np, grid_num, ray_min, ray_max, H, W, device)


    import copy
    model = copy.deepcopy(model).cpu()
    ray = ray.cpu()

    # ONNX 변환
    torch.cuda.empty_cache()
    model.eval()

    torch.onnx.export(
        model,                                      # 모델 (FG + BG)
        ray,                                        # 모델 입력 (튜플 또는 텐서)
        output_path,                                # 저장 경로
        export_params=True,                         # 모델 파라미터 저장 여부
        opset_version=12,                           # ONNX 버전
        do_constant_folding=True,                   # 상수 폴딩 최적화 여부
        input_names=['ray'],                        # 입력 이름
        output_names=['rgb'],                       # 출력 이름
        verbose=False,
        dynamic_axes={'ray': {0: 'batch_size'},
                      'rgb': {0: 'batch_size'}}
        # dynamic_axes=None
    )
    
    print(f"ONNX model saved to {output_path}")
    # model = onnx.load(output_path)
    # optimized_model = onnxoptimizer.optimize(model)
    # onnx.save(optimized_model, 'model.onnx')
    
    return ray


def convert_to_onnx_fg(model, output_path, H, W, device, mask_path=None):
    '''
    1. PyTorch에서 배경 제거 확인
    2. ONNX 변환
    '''
    # 입력 생성
    ray = get_test_rays(H, W).to(device)
    print(f"[FG] Input shape: {ray.shape}")

    if mask_path and os.path.exists(mask_path):
        mask_2d = torch.load(mask_path, map_location=device)    # [H, W]
        print(f"[FG] Loaded mask: {mask_2d.shape}, fg_ratio={mask_2d.mean():.3f}")

        if mask_2d.shape != (H, W):
            mask_2d = F.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='nearest'                            
            ).squeeze()

        mask_flat = mask_2d.view(-1, 1).to(device)  # [N, 1]
    
    else:
        mask_flat = torch.ones(ray.shape[0], 1, device=device)  # 마스크 파일 없으면 기존 설정 사용

    # 1
    model.eval()
    with torch.no_grad():
        rgb = model(ray, global_step='debug')  # 원본 모델

        imageio.imwrite("torch_test.png", to8b(rgb.detach().reshape(H, W, 3).cpu().numpy()))
        print("[FG] PyTorch test image saved to torch_test.png")

        rgb_masked = rgb * mask_flat           # 마스크 적용
        
        # 배경 색상 설정
        # bg_color = [1.0, 1.0, 1.0]
        # bg_color_tensor = torch.tensor(bg_color, device=device).view(1, 3)
        # rgb_masked = rgb * mask_flat + bg_color_tensor * (1 - mask_flat)

        # NumPy 변환
        rgb_np = rgb_masked.detach().cpu().numpy()
        rgb_image = rgb_np.reshape(H, W, 3)

        # 배경 픽셀 계산
        threshold = 0.01
        bg_pixels = (rgb_image.sum(axis=2) < threshold)
        bg_ratio = bg_pixels.sum() / (H*W)

        print(f"[FG] PyTorch 배경 제거 확인 결과]")
        print(f"[FG] 모델 픽셀: {(1-bg_ratio)*100:.1f}%")
        print(f"[FG] RGB 범위: [{rgb_np.min():.3f}, {rgb_np.max():.3f}]")
        
    # 2
    # ONNX export

    ''' 마스크 포함 ONNX 변환
    '''
    # class ModelWithMask(torch.nn.Module):
    #     def __init__(self, base_model, mask, bg_color=None):
    #         super().__init__()
    #         self.model = base_model
    #         self.register_buffer('mask', mask)  # [N, 1]
    #         # self.register_buffer('bg_color', torch.tensor(bg_color).view(1, 3))  # [1, 3]
        
    #     def forward(self, ray):
    #         rgb = self.model(ray)   # [N, 3]
    #         rgb_masked = rgb * self.mask    # 마스크 적용

    #         # 배경 색상 설정
    #         # rgb_masked = rgb * self.mask + self.bg_color * (1 - self.mask)
    #         return rgb_masked
    
    # model_with_mask = ModelWithMask(model, mask_flat).cpu().eval()
    # # model_with_mask = ModelWithMask(model, mask_flat, bg_color).cpu().eval()
    # ray_cpu = ray.cpu()

    ''' 마스크 제외 원본 모델만 ONNX 변환
    '''
    model_cpu = model.cpu().eval()
    ray_cpu = ray.cpu()

    torch.onnx.export(
        # model_with_mask,
        model_cpu,
        ray_cpu,
        output_path,
        export_params=True, 
        opset_version=12, 
        do_constant_folding=True,
        input_names=['ray'], 
        output_names=['rgb'], 
        # dynamic_axes=None,
        dynamic_axes={'ray': {0: 'N'},
                      'rgb': {0: 'N'}}
    )
    print(f"ONNX model saved to {output_path}")

    return ray_cpu

def generate_fg_image_from_model(model, occ_np, grid_num, ray_min, ray_max, H, W, device):
    # 1) 카메라 view의 모든 ray 생성
    ray = get_test_rays(H, W).to(device)   # [H*W, 4]
    ray_cpu = ray.cpu().numpy()

    # 2) 모델 추론
    with torch.no_grad():
        rgb = model(ray, global_step='debug')    # [H*W,3]
    rgb_np = rgb.cpu().numpy().reshape(H, W, 3)
    rgb_np = np.clip(rgb_np, 0.0, 1.0)

    # ---- 픽셀별 Foreground Mask 생성 ----
    Nu, Nv, Ns, Nt = grid_num
    s0, t0, u0, v0 = ray_min
    s1, t1, u1, v1 = ray_max

    # (H*W,) float arrays
    s = ray_cpu[:, 0]
    t = ray_cpu[:, 1]
    u = ray_cpu[:, 2]
    v = ray_cpu[:, 3]

    # float → voxel index
    def to_idx(val, v_min, v_max, N):
        idx = (val - v_min) / (v_max - v_min + 1e-8) * N
        idx = np.floor(idx).astype(np.int32)
        return np.clip(idx, 0, N - 1)

    u_idx = to_idx(u, u0, u1, Nu)
    v_idx = to_idx(v, v0, v1, Nv)
    s_idx = to_idx(s, s0, s1, Ns)
    t_idx = to_idx(t, t0, t1, Nt)

    # voxel occupancy 조회
    fg_mask_flat = occ_np[u_idx, v_idx, s_idx, t_idx]      # (H*W,)
    fg_mask = fg_mask_flat.reshape(H, W).astype(np.float32)

    print("ray_min:", ray_min)
    print("ray_max:", ray_max)
    print("s range:", float(s.min()), float(s.max()))
    print("t range:", float(t.min()), float(t.max()))
    print("u range:", float(u.min()), float(u.max()))
    print("v range:", float(v.min()), float(v.max()))
    print("FG count:", fg_mask_flat.sum(), "/", len(fg_mask_flat))

    # ---- 전경 RGB 생성 ----
    rgb_fg = rgb_np.copy()
    rgb_fg[fg_mask == 0] = 0.0    # 배경 → 0으로 제거함

    # ---- RGBA 생성 ----
    alpha = (fg_mask * 255).astype(np.uint8)       # 전경 255 / 배경 0
    rgba = np.concatenate([(rgb_fg * 255).astype(np.uint8), alpha[..., None]], axis=-1)

    # ---- 저장 ----
    # imageio.imwrite("foreground_view.png", (rgb_fg * 255).astype(np.uint8))
    # imageio.imwrite("foreground_view_rgba.png", rgba)

    # print("[OK] Saved foreground_view.png")
    # print("[OK] Saved foreground_view_rgba.png (transparent background)")


def test_onnx_model(onnx_path, ray, H, W, test_image_path):
    ''' ONNX 모델 테스트
    '''
    # ONNX 런타임 세션 생성
    print(f"Testing ONNX model from {onnx_path}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # GPU 우선, CPU는 백업
    ort_session = ort.InferenceSession(onnx_path, providers=providers)

    # 입력 데이터 준비
    ort_inputs = {
        'ray': ray.cpu().numpy()  # ONNX Runtime은 numpy 입력을 기대
    }
    
    # 렌더링 시간 측정
    render_time0 = torch.cuda.Event(enable_timing=True)
    render_time1 = torch.cuda.Event(enable_timing=True)
    render_time0.record()
    
    # ONNX 모델 추론
    ort_outputs = ort_session.run(None, ort_inputs)
    
    render_time1.record()
    torch.cuda.synchronize()
    inference_time = torch.cuda.Event.elapsed_time(render_time0, render_time1)
    
    ort_output = ort_outputs[0]
    np.save('onnx_output.npy', ort_output) # npy 저장
    rgb_onnx = ort_output.reshape(H, W, 3)
    imageio.imwrite(test_image_path, to8b(rgb_onnx))
    print(f"ONNX test image saved to {test_image_path}")
    print(f"ONNX 모델 렌더링 시간: {inference_time}ms")

def main():
    parser = config_parser()
    args = parser.parse_args()
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(args.gpuid)
        print('>>> Using GPU: {}'.format(args.gpuid))
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # torch.set_default_tensor_type('torch.cuda.HalfTensor')
    else:
        print('>>> Using CPU')
        
    # 모델 로드
    # model = load_model(args.ckpt_path)
    # model = model.to(device)

    # 원본 모델 로드
    model = load_model(
        args.ckpt_path,
        use_occ_mask=False,
        occ_tau=args.occ_tau,
        occ_reduce=args.occ_reduce,
        bg_val=args.bg_val
    ).to(device).eval()

    # if args.occ_mask_path is not None and os.path.exists(args.occ_mask_path):
    #     print(f"Loading occ mask from {args.occ_mask_path}")
    #     mask_4d = torch.load(args.occ_mask_path, map_location='cpu')  # [T,S,V,U]
    #     print(f"Mask shape: {mask_4d.shape}, range: [{mask_4d.min()}, {mask_4d.max()}]")
        
    #     model.set_occ_mask_4d(mask_4d)  # grid 마지막 채널에 occ mask 추가

    if args.occ_mask_path is not None:
        print("[WARN] --occ_mask_path is ignored")

    dataset_name = os.path.basename(args.ckpt_path).split('_')[0]
    decomp_name = os.path.basename(args.ckpt_path).split('_')[1]
    
    # 테스트 이미지 크기 설정
    if dataset_name == "beans":
        H = 512
        W = 512
    elif dataset_name == "bracelet":
        H = 512
        W = 256
    elif dataset_name == "gem":
        H = 384
        W = 512
    elif dataset_name == "truck":
        H = 640
        W = 480
    elif dataset_name == "chess":
        H = 700
        W = 400
    elif dataset_name == "bulldozer":
        H = 768
        W = 576
    elif dataset_name == "flowers":
        H = 640
        W = 768
    elif dataset_name == "treasure":
        H = 768
        W = 640
    else:
        H = 512
        W = 512
    
    # ONNX 변환
    ray = convert_to_onnx(
        model,
        args.output_path,
        H,
        W,
        device,
        mask_path='fg_uvst_table.pt'
    )
    
    # ONNX 모델 테스트 및 이미지 생성
    try:
        test_onnx_model(
            args.output_path, 
            ray, 
            H, 
            W,
            args.test_image_path
        )
    except ImportError:
        print("onnxruntime not installed. Skipping ONNX model testing.")
        print("Install with: pip install onnxruntime")

if __name__ == "__main__":
    main() 