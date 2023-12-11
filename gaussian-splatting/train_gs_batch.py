import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu_id', type=int, default=0)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=1)
# parser.add_argument('-n', '--note', type=str, default="limit-pts=4096-max10000-Dense")
args = parser.parse_args()

# read all objaverse ids
objv_data_root = "/nfs/ycheng/hyper-gaussian-splatting/data_objaverse_render"
all_objv_ids = os.listdir(objv_data_root)
all_objv_ids.sort()

sub_objv_ids = all_objv_ids[args.start_index:args.end_index]

print("Total number of objaverse ids: %d" % len(all_objv_ids))
print("Objaverse ids to be trained: %s" % str(sub_objv_ids))

for objv_id in sub_objv_ids:
    print("Training objaverse id: %s" % objv_id)
    cmd = f"python train_gs_batch.py -g {args.gpu_id} -o {objv_id}"
    print(cmd)
    os.system(cmd)

# setup output path


objv_id="0b1b44a101734a418455f738272e51a0"
# dset_path="../hyper-gaussian-splatting/data_objaverse_render/$objv_id"
dset_path="../data_objaverse_render/$objv_id"

# note="limit-pts=10000-noDense"
# note="limit-pts=4096-max10000-Dense"
# model_path="output/$objv_id-$note"

# save_root="output"
save_root="output_1211"
model_path="output/$objv_id-$note"

cmd="train.py -s $dset_path -m $model_path"

echo $cmd
CUDA_VISIBLE_DEVICES=$gpu_id python $cmd