import os

import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=1)
parser.add_argument('--port', type=int, default=6009)
# parser.add_argument('-n', '--note', type=str, default="limit-pts=4096-max10000-Dense")
args = parser.parse_args()

# read all objaverse ids
objv_data_root = "/nfs/ycheng/hyper-gaussian-splatting/data_objaverse_render"
all_objv_ids = os.listdir(objv_data_root)
all_objv_ids.sort()

sub_objv_ids = all_objv_ids[args.start_index:args.end_index]

print("Total number of objaverse ids: %d" % len(all_objv_ids))
print("Objaverse ids to be trained: %s" % len(sub_objv_ids))

# setup output path
gs_out_root="output_1211"
if not os.path.exists(gs_out_root):
    os.mkdir(gs_out_root)


for ix, objv_id in tqdm(enumerate(sub_objv_ids), total=len(sub_objv_ids)):

    # objv_id="0b1b44a101734a418455f738272e51a0"
    dset_path = f"../data_objaverse_render/{objv_id}"
    # model_path = "output/$objv_id-$note"
    model_path = f"{gs_out_root}/{objv_id}"

    cmd = f"CUDA_VISIBLE_DEVICES={args.gpu_id} python train.py -s {dset_path} -m {model_path} --port {args.port}"
    print(cmd)
    os.system(cmd)