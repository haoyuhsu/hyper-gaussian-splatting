
# objv_id="1ec7f"
objv_id="0a61021cae7e44208585d4c738904aca"
path="../hyper-gaussian-splatting/data_objaverse_render/$objv_id"
model_path="output/1ec7f-limit-pts=10000-noDense"
model_path="output/1ec7f-limit-pts=4096-max10000-Dense"


./SIBR_viewers/install/bin/SIBR_gaussianViewer_app --iteration 30000 --m ${model_path} --path ${path}