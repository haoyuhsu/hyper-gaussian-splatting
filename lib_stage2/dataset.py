import torch
from torch.utils.data import Dataset
import numpy as np
import os
import tqdm
from plyfile import PlyData, PlyElement

class GaussianPrimitivesDataset(Dataset):
    def __init__(self, root_dir: str, sh_degree: int, max_points: int = 10000):
        self.root_dir = root_dir
        self.max_sh_degree = sh_degree
        self.max_points = max_points
        
        all_scene_name_list = sorted(os.listdir(root_dir))
        self.scene_name_list = []
        skip_cnt = 0
        for scene_name in all_scene_name_list:
            ply_path = os.path.join(root_dir, scene_name, 'point_cloud', 'iteration_30000', 'point_cloud.ply')
            if os.path.exists(ply_path):
                self.scene_name_list.append(scene_name)
            else:
                print(f'Warning: {ply_path} does not exist. Skip this scene.')
                skip_cnt += 1
                
        print("Total number of scenes:", len(self.scene_name_list))
        print("Number of skipped scenes:", skip_cnt)

    def __len__(self):
        return len(self.scene_name_list)

    def __getitem__(self, idx: int):
        scene_name = self.scene_name_list[idx]
        scene_dir = os.path.join(self.root_dir, scene_name)

        # Load gaussian primitives
        plydata_path = os.path.join(scene_dir, 'point_cloud', 'iteration_30000', 'point_cloud.ply')
        plydata = PlyData.read(plydata_path)
        
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        assert xyz.shape[0] == opacities.shape[0] == features_dc.shape[0] == features_extra.shape[0] == scales.shape[0] == rots.shape[0]

        # randomly remove points or duplicate points to make the number of points = max_points
        if xyz.shape[0] > self.max_points:
            idx = np.random.choice(xyz.shape[0], self.max_points, replace=False)
            xyz = xyz[idx]
            opacities = opacities[idx]
            features_dc = features_dc[idx]
            features_extra = features_extra[idx]
            scales = scales[idx]
            rots = rots[idx]
        elif xyz.shape[0] < self.max_points:
            idx = np.random.choice(xyz.shape[0], self.max_points - xyz.shape[0], replace=True)
            xyz = np.concatenate((xyz, xyz[idx]), axis=0)
            opacities = np.concatenate((opacities, opacities[idx]), axis=0)
            features_dc = np.concatenate((features_dc, features_dc[idx]), axis=0)
            features_extra = np.concatenate((features_extra, features_extra[idx]), axis=0)
            scales = np.concatenate((scales, scales[idx]), axis=0)
            rots = np.concatenate((rots, rots[idx]), axis=0)

        # TODO: Load camera parameters
        # camera_params_path = os.path.join(scene_dir, 'cameras.json')
        
        ret = {
            'xyz': xyz,
            'opacities': opacities,
            'features_dc': features_dc,
            'features_extra': features_extra,
            'scales': scales,
            'rots': rots
        }
        return ret

        # return xyz, opacities, features_dc, features_extra, scales, rots
    
if __name__ == '__main__':
    gs_dataset = GaussianPrimitivesDataset(
        root_dir='/home/max/Downloads/tnrky8no4nv6d9bje805iyan3tb1a5fv (1).zip/output_1211/',
        sh_degree=3
    )
    dataloader = torch.utils.data.DataLoader(
        gs_dataset,
        batch_size=5,
        shuffle=False,
        num_workers=0
    )
    # test iteration of the dataloader
    for i, data in enumerate(dataloader):
        xyz, opacities, features_dc, features_extra, scales, rots = data
        print('xyz shape:', xyz.shape)
        print('opacities shape:', opacities.shape)
        print('features_dc shape:', features_dc.shape)
        print('features_extra shape:', features_extra.shape)
        print('scales shape:', scales.shape)
        print('rots shape:', rots.shape)
        break