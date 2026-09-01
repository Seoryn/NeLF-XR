import numpy as np
import os
from .load_llff import load_llff_data

def load_data(args):
    K, depths = None, None
    near_clip = None
    focal_depth = None
    movie_render_kwargs={
        'scale_r': 1.0, # circling radius
        'scale_f': 1.0, # the distance to the looking point of foucs
        'zdelta': 0.5,  # amplitude of forward motion
        'zrate': 1.0,   # frequency of forward motion
        'N_rots': 1,    # number of rotation in 120 frames
    }
    bd_factor = 0.75
    recenter = False
    print(f'Using LF data, bd_factor={bd_factor}, recenter={recenter}')
    if args.dataset_type == 'llff':
        images, depths, poses, bds, render_poses, i_test, focal_depth = load_llff_data(
                args.datadir, args.factor,
                recenter=recenter, bd_factor=bd_factor,
                width=None, height=None,
                spherify=False,
                load_depths=False,
                movie_render_kwargs=movie_render_kwargs)
        
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            if args.dataset_name == 'stanford':
                last_part = os.path.basename(args.datadir)
                sfx = '_{}'.format(args.factor) \
                    if args.factor != 1 else ''
                imgdir = os.path.join(args.datadir, 'images' + sfx)
                imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
                camera_coords = []
                for i in imgfiles:
                    if last_part in ['beans', 'tarot', 'tarot_small', 'knights']:
                        yx = i.split('_')[-2:]
                        y = float(yx[0])
                        x = float(yx[1].split('.png')[0])
                    else:
                        yx = i.split('_')[-3:-1]
                        y, x = -float(yx[0]), float(yx[1])
                    camera_coords.append((x, y, 0))
                camera_coords = np.array(camera_coords)
                x_range = (np.min(camera_coords[:, 0]), np.max(camera_coords[:, 0]))
                y_range = (np.min(camera_coords[:, 1]), np.max(camera_coords[:, 1]))
                # aspect = (x_range[1] - x_range[0]) / (y_range[1] - y_range[0])
                norm_x = ((camera_coords[:, 0] - x_range[0]) / (x_range[1] - x_range[0])) * 2 - 1
                norm_y = (((camera_coords[:, 1] - y_range[0]) / (y_range[1] - y_range[0])) * 2 - 1)
                # norm_y = (((camera_coords[:, 1] - y_range[0]) / (y_range[1] - y_range[0])) * 2 - 1) / aspect
                # norm_x = norm_x * 0.01
                # norm_y = norm_y * 0.01
                camera_coords = np.stack([norm_x, norm_y, np.zeros_like(norm_x)], axis=1) 
                
                x = np.linspace(-1, 1, 17)
                y = np.linspace(-1, 1, 17)
                xv, yv = np.meshgrid(x, y)
                camera_coords = np.stack([xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten())], axis=1)
                poses[:, :3, 3] = camera_coords
                
                # subsampled = []
                # for i in range(0, 17, 4):
                #     for j in range(0, 17, 4):
                #         index = i * 17 + j
                #         subsampled.append(index)
                # i_train = np.array([i for i in subsampled])
                # i_test = np.array([i for i in np.arange(int(images.shape[0])) if
                #                 (i not in i_train)])
                # i_val = i_test
                # print('Subsampled train', i_train)
                # print('Subsampled test', i_test)
                
                i_test = np.array([72, 76, 80, 140, 144, 148, 208, 212, 216])
                i_val = i_test
                i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                                (i not in i_test and i not in i_val)])
                
                # i_test = np.arange(images.shape[0])[::args.llffhold]
                # i_val = i_test
                # i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                #                 (i not in i_test and i not in i_val)])
                
                print('i_val', i_val)
                print('i_train', i_train)

            elif args.dataset_name == '4by4':
                val = []
                for i in range(41, 80, 16):
                    for j in range(8):
                        val.append(j+i)
                for i in range(97, 140, 16):
                    for j in range(8):
                        val.append(j+i)
                i_val = np.array(val)
                i_test = i_val
                i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                                (i not in i_test)])
            else:
                print('Auto LLFF holdout,', args.llffhold)
                i_test = np.arange(images.shape[0])[::args.llffhold]
                i_val = i_test
                i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                                (i not in i_test and i not in i_val)])
        
        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        else:
            near_clip = max(np.ndarray.min(bds) * .9, 0)
            _far = max(np.ndarray.max(bds) * 1., 0)
            near = 0
            far = inward_nearfar_heuristic(poses[i_train, :3, 3])[1]
            print('near_clip', near_clip)
            print('original far', _far)
        print('NEAR FAR', near, far)
    
    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]
    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks,
        near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,
        irregular_shape=irregular_shape,
        focal_depth=focal_depth,
    )
    return data_dict

def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far