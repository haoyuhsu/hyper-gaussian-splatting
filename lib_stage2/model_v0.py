

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.vae import VAE

from pytorch3d.loss import chamfer_distance

# torchvision
import torchvision.utils as vutils

# GS
import sys
sys.path.append("./gaussian-splatting")

from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamSimple
from utils.camera_utils import loadCamSimple
from gaussian_renderer import render_simple


def set_gaussians_ply(gaussians, ply_dict):

    xyz = ply_dict['xyz']
    features_dc = ply_dict['features_dc']
    features_extra = ply_dict['features_extra']
    opacities = ply_dict['opacities']
    scales = ply_dict['scales']
    rots = ply_dict['rots']

    # gaussians.set_xyz(xyz)
    # gaussians.set_rgb(rgb)
    # gaussians.set_normals(normals)
    # gaussians.set_opacities(opacities)
    # gaussians.set_scales(scales)
    # gaussians.set_rots(rots)

    # gaussians._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    # gaussians._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # gaussians._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # gaussians._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    # gaussians._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    # gaussians._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    gaussians._xyz = nn.Parameter(xyz.clone().requires_grad_(True))
    gaussians._features_dc = nn.Parameter(features_dc.clone().requires_grad_(True))
    gaussians._features_rest = nn.Parameter(features_extra.clone().requires_grad_(True))
    gaussians._opacity = nn.Parameter(opacities.clone().requires_grad_(True))
    gaussians._scaling = nn.Parameter(scales.clone().requires_grad_(True))
    gaussians._rotation = nn.Parameter(rots.clone().requires_grad_(True))


class GenGSModel:
    
    def __init__(self, args):
        
        
        self.args = args
        self.device = args.device
        
        self.model = VAE(args)

        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.init_renderer()


    def init_renderer(self):

        sh_degree = 3 # hardcoded
        self.gaussians = GaussianModel(sh_degree)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        # load a random camera
        ## camera
        # 1. load camera info
        objv_id = "0b1b44a101734a418455f738272e51a0"
        transforms_path = f"data_objaverse_render/{objv_id}/transforms_train.json"
        self.cam_info_list = readCamSimple(transforms_path)

        # 2. load cam model from cam info
        self.cam_list = []
        for id, c in enumerate(self.cam_info_list):
            camera = loadCamSimple(c, id, device=self.device)
            self.cam_list.append(camera)


    def render(self, ply_dict, cam=None):

        if cam is None:
            cam = self.cam_list[0]

        B = ply_dict['xyz'].shape[0]

        # bs = 1 first
        # assert ply_dict['xyz'].shape[0] == 1

        rend_list = []  

        # replacye gs_ply to gaussians
        for bi in range(B):
            ply_dict_bi = {}
            for k, v in ply_dict.items():
                cur_v = v[bi]

                # transpose if _features_dc or _features_extra
                if k == 'features_dc' or k == 'features_extra':
                    cur_v = cur_v.transpose(1, 2).contiguous()

                ply_dict_bi[k] = cur_v

            set_gaussians_ply(self.gaussians, ply_dict_bi)

            ret = render_simple(cam, self.gaussians, self.background)
            rendering = ret["render"]

            rend_list.append(rendering)

            # gt = cam0.original_image[0:3, :, :]
            # vutils.save_image(rendering, f'tmp/rend-recon-{bi}.png')
            # vutils.save_image(gt, 'gt.png')

        rend = torch.stack(rend_list, dim=0)
        return rend

    def set_input(self, input):

        # send all element to device
        for k, v in input.items():
            input[k] = v.to(self.args.device)
        
    def train_one_batch(self, data):

        self.model.train()
        
        # ply_dict = data.copy()
        # rend = self.render(ply_dict)
        # vutils.save_image(rend, 'tmp/rend.png')
        # import pdb; pdb.set_trace()
        
        dec_dict, ret_dict = self.model(data)
        
        # loss
        pred_xyz = dec_dict["xyz"]
        gt_xyz = data["xyz"]
        
        pred_opacities = dec_dict["opacities"]
        gt_opacities = data["opacities"]
        
        pred_features_dc = dec_dict["features_dc"]
        gt_features_dc = data["features_dc"]
        
        pred_features_extra = dec_dict["features_extra"]
        gt_features_extra = data["features_extra"]
        
        pred_scales = dec_dict["scales"]
        gt_scales = data["scales"]
        
        pred_rots = dec_dict["rots"]
        gt_rots = data["rots"]
        
        # chamfer for pred_xyz, gt_xyz
        loss_xyz, _ = chamfer_distance(pred_xyz, gt_xyz)
        
        loss_opacities = F.mse_loss(pred_opacities, gt_opacities) * self.args.opac_weight
        loss_sh_dc = F.l1_loss(pred_features_dc, gt_features_dc)
        loss_sh_extra = F.l1_loss(pred_features_extra, gt_features_extra)
        
        loss_scales = F.mse_loss(pred_scales, gt_scales) * self.args.scale_weight
        loss_rots = F.mse_loss(pred_rots, gt_rots)

        # kl divergence loss
        loss_kl = -0.5 * torch.sum(1 + ret_dict['logvar'] - ret_dict['mu'].pow(2) - ret_dict['logvar'].exp())
        loss_kl *= self.args.kl_weight

        loss = loss_xyz + loss_opacities + \
               loss_sh_dc + loss_sh_extra + \
               loss_scales  + loss_rots + loss_kl
        
        loss_dict = {
            'loss_all': loss.item(), 'loss_kl': loss_kl.item(),
            'loss_xyz': loss_xyz.item(), 'loss_opac': loss_opacities.item(),
            'loss_sh_dc': loss_sh_dc.item(), 'loss_sh_ex': loss_sh_extra.item(), 'loss_scales': loss_scales.item(),
            'loss_rots': loss_rots.item(),
        }
        self.loss_dict = loss_dict
        
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    @torch.no_grad()
    def recon(self, data):
        
        self.model.eval()
        dec_dict, ret_dict = self.model(data)
        rend_recon = self.render(dec_dict)

        rend_gt = self.render(data)
        # vutils.save_image(rend, 'tmp/rend.png')
        return rend_recon, rend_gt
    
    @torch.no_grad()
    def inference(self, bs=16):
        
        self.model.eval()
        z = torch.randn(bs, self.args.z_dim).to(self.args.device)
        dec_dict = self.model.decode(z)

        # save as ply
        pred_rend = self.render(dec_dict)

        return pred_rend

    
    def save_ckpt(self, save_path):
        
        # save checkpoint
        sd = {
                'model': self.model.cpu().state_dict(),
                'iters': self.args.cur_iter,
            }
        self.model.to(self.args.device)
            
        torch.save(sd, save_path)
        print(f'saving model to {save_path}')