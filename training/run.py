import os, time, random, argparse
from tqdm import tqdm, trange
import imageio
import numpy as np
import cv2
import random
import torch
import torch.nn.functional as F
# from torch.cuda.amp import autocast, GradScaler

from lib import utils,dlfgo_model
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
    parser.add_argument("--grid_num", type=int, nargs='+', default=0, help="A list of integers to process")
    parser.add_argument("--grid_dim", type=int, default=16, help='color voxel grid dimension')
    parser.add_argument("--mlp_depth", type=int, default=4, help='color voxel grid depth')
    parser.add_argument("--mlp_width", type=int, default=128, help='color voxel grid width')
    parser.add_argument("--lr_grid", type=float, default=1e-01, help='lr of grid')
    parser.add_argument("--wd_grid", type=float, default=0, help='lr weight decay of grid')
    parser.add_argument("--lr_mlp", type=float, default=1e-03, help='lr of the mlp')
    parser.add_argument("--batch_size", type=int, default=8192, help='batch size (number of random rays per optimization step)')
    parser.add_argument("--load2gpu_on_the_fly", action='store_true',)
    parser.add_argument("--basedir", type=str, default='/home/seoryn/onnx/bulldozer_large', help='base directory')
    parser.add_argument("--expname", type=str, default='', help='experiment name')
    # parser.add_argument("--datadir", type=str, default='/data/igjeong/dataset/stanford_half/truck', help='data directory')
    parser.add_argument("--datadir", type=str, default='/data/seoryn/data/bulldozer')
    parser.add_argument("--factor", type=int, default=1, help='downsample factor for LLFF data')
    parser.add_argument("--llffhold", type=int, default=8, help='holdout for LLFF data')
    parser.add_argument("--dataset_type", type=str, default='llff', help='dataset type')
    parser.add_argument("--dataset_name", type=str, default='stanford', help='dataset name')
    parser.add_argument("--epoch", default=1000, type=int)
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--pe",type=int, default=0)
    parser.add_argument("--decomp", type=str, default='4d')
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument("--interp_mode", type=int, default=16)
    # logging/saving options
    parser.add_argument("--save" , action='store_true')
    # parser.add_argument("--i_weights", type=int, default=1000000,
    #                     help='frequency of weight ckpt saving')
    parser.add_argument("--save_epoch", type=int, default=250,
                        help='frequency of model saving')
    return parser

@torch.no_grad()
def warmup(model, ray_type, H, W, K, c2w):
    '''Warm up the model for faster inference.
    '''
    if ray_type == 'twoplane':
        ray = utils.get_rays_of_a_view_twoplane(H, W, K, c2w)
    # elif ray_type == 'plucker':
    #     ray = utils.get_rays_of_a_view_plucker(H, W, K, c2w)
    else:
        raise ValueError(f'Unknown ray_type: {ray_type}')
    _ = model(ray).reshape(H,W,-1)
    del _

@torch.no_grad()
def render_viewpoints(model, ray_type, render_set, render_poses, HW, Ks,
                        gt_imgs=None, savedir=None, dump_images=True, expdir=None, render_factor=0,
                        eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor
        
    rgbs = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    render_time = []
    H, W = HW[0]
    K = Ks[0]
    # focal_depth = torch.tensor([focal_depth], dtype=torch.float32)
    
    # warm up
    warmup(model, ray_type, H, W, K, render_poses[0])
 
    if ray_type == 'twoplane':
        # H = 1024
        # W = 1024
        # K = torch.tensor([[H, 0, H/2], [0, H, W/2], [0, 0, 1]], dtype=torch.float32)
        
        for i, c2w in enumerate(tqdm(render_poses)):
            render_time0 = torch.cuda.Event(enable_timing=True)
            render_time1 = torch.cuda.Event(enable_timing=True)
            c2w = torch.Tensor(c2w)
            ray = utils.get_rays_of_a_view_twoplane(H, W, K, c2w)
            render_time0.record()
            render_result = model(ray).reshape(H,W,-1)
            render_time1.record()
            torch.cuda.synchronize()
            render_time.append(torch.cuda.Event.elapsed_time(render_time0, render_time1))
            rgb = render_result.cpu().numpy()
            rgbs.append(rgb)
            if i==0:
                print('Testing', rgb.shape)
            if gt_imgs is not None and render_factor==0 and i < len(gt_imgs):
                p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                psnrs.append(p)
                if eval_ssim:
                    ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
                if eval_lpips_alex:
                    lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
                if eval_lpips_vgg:
                    lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))
                    
    if len(psnrs):
        avg_psnr = np.mean(psnrs)
        test_str = f'Set: {render_set} / Testing psnr: {avg_psnr} (avg) \n'
        if eval_ssim: test_str += f' / Testing ssim: {np.mean(ssims)} (avg) \n'
        if eval_lpips_vgg: test_str += f' / Testing lpips (vgg): {np.mean(lpips_vgg)} (avg) \n'
        if eval_lpips_alex: test_str += f' / Testing lpips (alex): {np.mean(lpips_alex)} (avg) \n'
        if len(render_time): test_str += f' / Testing render time: {np.mean(render_time)} (avg) '
        print(test_str)
        if expdir is not None:
            with open(f'{expdir}/test_results.txt', 'a') as f:
                f.write(test_str+'\n')
                
    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            # save rgb image
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            # save diff image
            # diff = rgbs[i] - gt_imgs[i]
            # diff8 = utils.to8b(abs(diff)*10)
            # heatmap = cv2.applyColorMap(diff8, cv2.COLORMAP_RAINBOW)
            # filename = os.path.join(savedir, '{:03d}_diff.png'.format(i))
            # imageio.imwrite(filename, heatmap)
    rgbs = np.array(rgbs)
    return rgbs
    

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
    kept_keys = {
            'hwf', 'HW', 'Ks', 'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images', 'focal_depth'}
    # remove useless field
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)
    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict

def print_cuda_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # 메모리 사용량 (MB)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # 예약된 메모리 (MB)
    print(f"Allocated memory: {allocated:.2f} MB / Reserved memory: {reserved:.2f} MB")
    
def create_new_model(args, ray_min, ray_max):
    print(f'\033[96muse LightField {args.ray_type} grid\033[0m')
    if args.ray_type == 'twoplane':        
        model = dlfgo_model.DLFGO_twoplane(
            ray_min=ray_min, ray_max=ray_max, grid_num=args.grid_num,
            ray_type='twoplane', grid_dim=args.grid_dim,
            mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
            pe=args.pe,
            decomp=args.decomp, levels=args.levels,
            interp_mode=args.interp_mode,)
    else:
        raise ValueError(f'Unknown ray_type: {args.ray_type}')
    model = model.to(device)
    optimizer = torch.optim.Adam([
        {'params': model.grid.parameters(), 'lr': args.lr_grid},
        {'params': model.mlp.parameters(), 'lr': args.lr_mlp}
    ])
    return model, optimizer

def load_existed_model(args, reload_ckpt_path):
    if args.ray_type == 'twoplane':
        model_class = dlfgo_model.DLFGO_twoplane
    # elif args.ray_type == 'plucker':
    #     model_class = dlfgo_model.DirectLF5DGO
    else:
        raise ValueError(f'Unknown ray_type: {args.ray_type}')
    model = utils.load_model(model_class, reload_ckpt_path).to(device)
    optimizer = torch.optim.Adam([
        {'params': model.grid.parameters(), 'lr': args.lr_grid},
        # {'params': model.encoder.parameters(), 'lr': args.lr_grid},
        {'params': model.mlp.parameters(), 'lr': args.lr_mlp}
    ])
    model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start

def scene_rep_reconstruction(args, data_dict):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HW, Ks, i_train, i_val, i_test, poses, render_poses, images, focal_depth = [
        data_dict[k] for k in [
            'HW', 'Ks', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'focal_depth'
        ]]
    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(args.basedir, args.expname, f'train_last.tar')
    
    if args.no_reload:
        reload_ckpt_path = None
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None
    
    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction: train from scratch')
        rgb_all = images[i_train].to('cpu' if args.load2gpu_on_the_fly else device)
        ray_min, ray_max = utils.get_ray_min_max(
            ray_type=args.ray_type, rgb_all=rgb_all,
            train_poses=poses[i_train], HW=HW[i_train], Ks=Ks[i_train])
        model, optimizer = create_new_model(args, ray_min, ray_max)
        start = 0
    else:
        print(f'scene_rep_reconstruction: reload from {reload_ckpt_path}')
        model, optimizer, start = load_existed_model(args, reload_ckpt_path)
    
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
    
    epoch = args.epoch
    iter_per_epoch = len(rgb_train) // args.batch_size
    iter = epoch * iter_per_epoch
    print(f'Training rays : {len(rgb_train)} / Batch size: {args.batch_size} / Epoch: {epoch} / Iter: {iter} / Iter per epoch: {iter_per_epoch}')
    
    torch.cuda.empty_cache()
    psnr_lst = []
    psnr_val_lst = []
    time0 = time.time()
    global_step = -1
    print("before training, cuda memory usage check")
    print_cuda_memory_usage()
    
    # wandb.init()
    
    pbar = trange(1+start, 1+iter)
    for global_step in pbar:
        # random sample rays
        selected_indices = batch_index_sampler()
        rgb_train_batch = rgb_train[selected_indices]
        ray_train_batch = ray_train[selected_indices]
        # viewdirs_train_batch = viewdirs_train[selected_indices]
            
        optimizer.zero_grad(set_to_none=True)
        
        # with autocast():
        # render_result = model(ray_train_batch, viewdirs_train_batch, global_step)
        render_result = model(ray_train_batch, global_step)
        # loss = F.l1_loss(render_result, rgb_train_batch)
        loss = F.mse_loss(render_result, rgb_train_batch)
        psnr = utils.mse2psnr(loss.detach())
        loss = loss * 1000
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        loss.backward()
        optimizer.step()
        psnr_lst.append(psnr.item())
        
        # update progress bar description with loss and psnr
        # pbar.set_description(f"Loss: {loss.item():.7f}")
        pbar.set_description(f"Loss: {loss.item():.7f}, PSNR: {psnr.item():.2f}")
        
        
        # update lr
        # decay_steps = iter_per_epoch
        decay_factor = 0.995 ** (1/iter_per_epoch)
        # decay_steps = args.lr_decay * 1000
        # decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # wandb.log() 

        # check log & save
        if global_step%(1*iter_per_epoch)==0 or (global_step%(1*iter_per_epoch)==0 and global_step < (10*iter_per_epoch)) or (global_step%2000==0 and global_step < (10*iter_per_epoch)):
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            write_str = f'expname / {args.expname} / '
            write_str += f'Eps / {eps_time_str} / '
            write_str += f'grid lr / {optimizer.param_groups[0]["lr"]:.8f} / '
            write_str += f'mlp lr / {optimizer.param_groups[1]["lr"]:.8f} / '
            write_str += f'Loss / {loss.item():.7f} / '
            write_str += f'global_step / {global_step:6d} / '
            write_str += f'epoch / {global_step//iter_per_epoch:4d} / Train PSNR / {np.mean(psnr_lst):5.2f} / '
            if global_step%(10*iter_per_epoch)==0 or (global_step%(1*iter_per_epoch)==0 and global_step < (10*iter_per_epoch)):
                with torch.no_grad():
                    if i_val is None:
                        psnr_val = None
                    else:
                        # render_result_val = model(ray_val, viewdirs_val, global_step)
                        render_result_val = model(ray_val, global_step)
                        mse_loss_val = F.mse_loss(render_result_val, rgb_val)
                        psnr_val = utils.mse2psnr(mse_loss_val.detach())
                        psnr_val_lst.append(psnr_val.item())
                        write_str += f'Val PSNR / {np.mean(psnr_val_lst):5.2f} / '
                # wandb.log()
            tqdm.write(write_str)
            with open(os.path.join(args.basedir, args.expname, 'training_result.txt'), 'a') as f:
                f.write(write_str+'\n')
            psnr_lst = []
            psnr_val_lst = []

        # if global_step%args.i_weights==0:
        #     path = os.path.join(args.basedir, args.expname, f'train_{global_step:06d}.tar')
        #     torch.save({
        #         'global_step': global_step,
        #         'model_kwargs': model.get_kwargs(),
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         # 'scaler_state_dict': scaler.state_dict(),
        #     }, path)
        #     print(f'scene_rep_reconstruction: saved checkpoints at', path)
        if global_step%(args.save_epoch*iter_per_epoch)==0:
            model_path = os.path.join(args.basedir, args.expname, 'model', f'train_last_{global_step//(iter_per_epoch)}epoch.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scaler_state_dict': scaler.state_dict(),
            }, model_path)
            print(f'scene_rep_reconstruction: saved checkpoints at', model_path)
    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scaler_state_dict': scaler.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction: saved checkpoints at', last_ckpt_path)
        weight_path = os.path.join(args.basedir, args.expname, f'train_last_weight.pt')
        torch.save({'model_kwargs': model.get_kwargs(),
                    'model_state_dict': model.state_dict(),}
                   , weight_path)
        print(f'scene_rep_reconstruction: saved weights at', weight_path)
    
        

def train(args, data_dict):
    torch.cuda.empty_cache()
    # init
    eps_time = time.time()
    os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
    os.makedirs(os.path.join(args.basedir, args.expname, 'model'), exist_ok=True)
    with open(os.path.join(args.basedir, args.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    
    scene_rep_reconstruction(args, data_dict)
    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')
    # wandb.finish()
    
if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    data_name = os.path.basename(args.datadir)
    if args.grid_num == 0:
        if data_name == "beans":
            args.grid_num = [8, 8, 128, 64]
            # args.grid_num = [4, 4, 128, 64]
        elif data_name == "bracelet":
            args.grid_num = [8, 8, 128, 80]
            # args.grid_num = [8, 8, 256, 160]
            # args.grid_num = [4, 4, 128, 80]
        elif data_name == "gem":
            args.grid_num = [8, 8, 96, 128]
            # args.grid_num = [4, 4, 96, 128]
        elif data_name == "truck":
            args.grid_num = [8, 8, 160, 120]
            # args.grid_num = [4, 4, 160, 120]
        elif data_name == "chess":
            args.grid_num = [8, 8, 175, 100]
            # args.grid_num = [4, 4, 175, 100]
        elif data_name == "bulldozer":
            args.grid_num = [8, 8, 192, 144]
            # args.grid_num = [4, 4, 192, 144]
        elif data_name == "flowers":
            args.grid_num = [8, 8, 160, 192]
            # args.grid_num = [4, 4, 160, 192]
        elif data_name == "treasure":
            args.grid_num = [8, 8, 192, 160]
            # args.grid_num = [4, 4, 192, 160]
            # args.grid_num = [8, 8, 116, 140]
        else:
            # args.grid_num = [8, 8, 512, 512]
            args.grid_num = [8, 8, 128, 128]
            # args.grid_num = [17, 17, 512, 512]
            
            # args.grid_num = [4, 4, 128, 128]
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(args.gpuid)
        print('>>> Using GPU: {}'.format(args.gpuid))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # torch.set_default_tensor_type('torch.cuda.HalfTensor')
    else:
        print('>>> Using CPU')
    seed_everything()
    # Load data
    data_dict = load_everything(args)
    if not args.render_only:
        train(args, data_dict)
        
    # load model for rendring
    if args.render_test or args.render_train:
        ckpt_path = os.path.join(args.basedir, args.expname, 'train_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if args.ray_type == 'twoplane':
            model_class = dlfgo_model.DLFGO_twoplane
        # elif args.ray_type == 'plucker':
        #     model_class = dlfgo_model.DirectLF5DGO
        else:
            raise ValueError(f'Unknown ray_type: {args.ray_type}')
        model = utils.load_model(model_class, ckpt_path).to(device)
    
    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(args.basedir, args.expname, f'render_train_{ckpt_name}')
        expdir = os.path.join(args.basedir, args.expname)
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs = render_viewpoints(
            model=model, ray_type=args.ray_type, render_set='train',
            render_poses=data_dict['poses'][data_dict['i_train']],
            HW=data_dict['HW'][data_dict['i_train']],
            Ks=data_dict['Ks'][data_dict['i_train']],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
            savedir=testsavedir, dump_images=args.dump_images, expdir=expdir,
            eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
    
    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(args.basedir, args.expname, f'render_test_{ckpt_name}')
        expdir = os.path.join(args.basedir, args.expname)
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs = render_viewpoints(
            model=model, ray_type=args.ray_type, render_set='test',
            render_poses=data_dict['poses'][data_dict['i_test']],
            HW=data_dict['HW'][data_dict['i_test']],
            Ks=data_dict['Ks'][data_dict['i_test']],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
            savedir=testsavedir, dump_images=args.dump_images, expdir=expdir,
            eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
    
    print('Done')