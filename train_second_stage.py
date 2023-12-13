import os
import sys
import argparse

import numpy as np
from tqdm import tqdm

# torch
import torch
from torch.utils.tensorboard import SummaryWriter

import torchvision.utils as vutils

from lib_stage2.dataset import GaussianPrimitivesDataset
from lib_stage2.model_v0 import GenGSModel

# GS
sys.path.append("./gaussian-splatting")

from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamSimple
from utils.camera_utils import loadCamSimple
from gaussian_renderer import render_simple


def test_render():

    sh_degree = 3 # hardcoded
    gaussians = GaussianModel(sh_degree)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # transforms_path = "./gaussian-splatting/output_1211/0b1b44a101734a418455f738272e51a0/cameras.json"

    objv_id = "0b1b44a101734a418455f738272e51a0"

    ## ply path
    ply_path = f"gaussian-splatting/output_1211/{objv_id}/point_cloud/iteration_30000/point_cloud.ply"
    gaussians.load_ply(ply_path)

    ## camera
    # 1. load camera info 
    transforms_path = f"data_objaverse_render/{objv_id}/transforms_train.json"
    cam_info_list = readCamSimple(transforms_path)

    # 2. load cam model from cam info
    cam_list = []
    for id, c in enumerate(cam_info_list):
        camera = loadCamSimple(c, id, device="cuda")
        cam_list.append(camera)

    # render 10 images
    rend_dir = f"paper_figs/rend/{objv_id}"
    if not os.path.exists(rend_dir): os.makedirs(rend_dir)

    nview = 5
    # rend_ixs = range(0, len(cam_list), len(cam_list) // nview)
    import numpy as np
    rend_ixs = np.random.choice(len(cam_list), nview, replace=False)
    rend_list = []
    for i in tqdm(rend_ixs, total=len(rend_ixs)):
        cam = cam_list[i]

        ret = render_simple(cam, gaussians, background)
        rendering = ret["render"]
        # vutils.save_image(rendering, f'{rend_dir}/rend_{i}.png')

        rend_list.append(rendering)

    rend_all = torch.cat(rend_list, dim=-1)
    vutils.save_image(rend_all, f'paper_figs/rend/{objv_id}.png')

def test_render_batch():

    sh_degree = 3 # hardcoded
    gaussians = GaussianModel(sh_degree)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # transforms_path = "./gaussian-splatting/output_1211/0b1b44a101734a418455f738272e51a0/cameras.json"
    

    # load cam
    
    ## camera
    # 1. load camera info 
    objv_id = "0b1b44a101734a418455f738272e51a0"
    transforms_path = f"data_objaverse_render/{objv_id}/transforms_train.json"
    cam_info_list = readCamSimple(transforms_path)

    # 2. load cam model from cam info
    cam_list = []
    for id, c in enumerate(cam_info_list):
        camera = loadCamSimple(c, id, device="cuda")
        cam_list.append(camera)


    objv_ids = os.listdir("gaussian-splatting/output_1211")
    cnt = 0
    for objv_id in objv_ids:
        if cnt > 100: break

        ## ply path
        ply_path = f"gaussian-splatting/output_1211/{objv_id}/point_cloud/iteration_30000/point_cloud.ply"
        if not os.path.exists(ply_path): continue
        gaussians.load_ply(ply_path)

        # render nview images
        # rend_dir = f"paper_figs/rend/{objv_id}"
        # if not os.path.exists(rend_dir): os.makedirs(rend_dir)

        nview = 5
        # rend_ixs = range(0, len(cam_list), len(cam_list) // nview)
        rend_ixs = np.random.choice(len(cam_list), nview, replace=False)
        rend_list = []
        for i in tqdm(rend_ixs, total=len(rend_ixs)):
            cam = cam_list[i]

            ret = render_simple(cam, gaussians, background)
            rendering = ret["render"]
            # vutils.save_image(rendering, f'{rend_dir}/rend_{i}.png')

            rend_list.append(rendering)

        rend_all = torch.cat(rend_list, dim=-1)
        vutils.save_image(rend_all, f'paper_figs/rend/{objv_id}.png')
        cnt += 1
        print(f"cnt: {cnt}")


    # cam0 = cam_list[0]

    # ret = render_simple(cam0, gaussians, background)
    # rendering = ret["render"]

    # gt = cam0.original_image[0:3, :, :]

    # vutils.save_image(rendering, 'rend.png')
    # vutils.save_image(gt, 'gt.png')

def infinite_iter(dataloader):
    while True:
        for data in dataloader:
            yield data

if __name__ == "__main__":
    # test_render()
    test_render_batch()
    0/0

    # import pdb; pdb.set_trace()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='stage2_model')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--niters', type=int, default=1000000)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # model
    parser.add_argument('--z_dim', type=int, default=1024)
    parser.add_argument('--kl_weight', type=float, default=1e-4)

    parser.add_argument('--opac_weight', type=float, default=1e-3)
    parser.add_argument('--scale_weight', type=float, default=1e-2)
    
    args = parser.parse_args()
    
    device = 'cuda'
    args.device = device

    # logdir
    exp_dir = args.exp_name

    img_dir = f'logs/{exp_dir}/images'
    ckpt_dir = f'logs/{exp_dir}/ckpts'
    tb_dir = f'logs/{exp_dir}/tb'
    all_d = [img_dir, ckpt_dir, tb_dir]
    for d in all_d:
        if not os.path.exists(d):
            os.makedirs(d)
    writer = SummaryWriter(tb_dir)
    
    # data
    data_root = './gaussian-splatting/output_1211'
    gs_dataset = GaussianPrimitivesDataset(
        # root_dir='/home/max/Downloads/tnrky8no4nv6d9bje805iyan3tb1a5fv (1).zip/output_1211/',
        root_dir=data_root,
        sh_degree=3
    )
    train_dataloader = torch.utils.data.DataLoader(
        gs_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    train_data_iter = infinite_iter(train_dataloader)
    
    model = GenGSModel(args)
    model.writer = writer
    
    total_iters = args.niters
    pbar = tqdm(range(total_iters), total=total_iters)
    
    for i in range(total_iters):
        args.cur_iter = i
        
        data = next(train_data_iter)
        
        model.set_input(data)
        
        model.train_one_batch(data)
        
        # print loss
        if i % 10 == 0:
            
            loss_str = f'[{i}/{total_iters}],'
            for k, v in model.loss_dict.items():
                # print(f"{k}: {v:.5f}")
                loss_str += f" {k}: {v:.5f},"
                writer.add_scalar(f'loss/{k}', v, i)
            
            pbar.set_description(loss_str)

        # render: recon
        if i % 5000 == 0:
            
            rend_recon, rend_gt = model.recon(data)
            rend_pred = model.inference(bs=args.batch_size)

            rend_gt = vutils.make_grid(rend_gt, nrow=4, normalize=True)
            rend_recon = vutils.make_grid(rend_recon, nrow=4, normalize=True)
            rend_pred = vutils.make_grid(rend_pred, nrow=4, normalize=True)

            writer.add_image('0-gt', rend_gt, i)
            writer.add_image('1-recon', rend_recon, i)
            writer.add_image('2-pred', rend_pred, i)
            
        # save model
        if i % 15000 == 0 or i == args.niters - 1:
            save_path = f'{ckpt_dir}/iter_{i}.pth'
            model.save_ckpt(save_path)
            
        pbar.update(1)
    
    
    
    
    
