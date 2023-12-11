

gpu_id=7
# objv_id="1ec7f"
# objv_id="0a61021cae7e44208585d4c738904aca"
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