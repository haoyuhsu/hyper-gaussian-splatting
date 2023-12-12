
gpu_id=1
objv_id="0a61021cae7e44208585d4c738904aca"
objv_id="0b1b44a101734a418455f738272e51a0"
# path="../hyper-gaussian-splatting/data_objaverse_render/$objv_id"
path="../data_objaverse_render/$objv_id"
# model_path="output/1ec7f-limit-pts=10000-noDense"
# model_path="output/$objv_id-limit-pts=4096-max10000-Dense"
model_path="output_1211/$objv_id"

# ./SIBR_viewers/install/bin/SIBR_gaussianViewer_app --iteration 30000 --m ${model_path} --path ${path}
CUDA_VISIBLE_DEVICES=$gpu_id python render.py -m $model_path -s $path