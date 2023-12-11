
objv_id="1ec7f"
dset_path="../hyper-gaussian-splatting/data_objaverse_render/$objv_id"

# note="limit-pts=10000-noDense"
note="limit-pts=4096-max10000-Dense"
model_path="output/$objv_id-$note"

cmd="train.py -s $dset_path -m $model_path"

echo $cmd

python $cmd