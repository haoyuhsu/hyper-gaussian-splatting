# hyper-gaussian-splatting

## Environment Setup

### Install Blender

#### Option 1: install Blender from website
- reference: https://www.blender.org/download/

#### Option 2: build Blender as a Python module 
- reference: https://wiki.blender.org/wiki/Building_Blender/Other/BlenderAsPyModule
- ***attempt once but failed.***

### Install Python packages
```
pip install objaverse
```
### Install nerfstudio
- reference: https://docs.nerf.studio/quickstart/installation.html

## Run Code

### Run dataset sampling
```
export HOME_DIR=$HOME
export DATA_ROOT_DIR=$HOME/Desktop/hyper-gaussian-splatting/data_objaverse/
export START_INDEX=0
export END_INDEX=50  # -1: whole list
export NUM_VIEWS=100  # number of views sampled per object asset

mkdir data_objaverse/
mkdir data_objaverse_render/

# download assets from Objaverse
python download_objaverse.py \
    --output_dir $DATA_ROOT_DIR \
    --start_index $START_INDEX \
    --end_index $END_INDEX

# find blender bin path
which blender  # return "/snap/bin/blender"

# render views using blender-python
/snap/bin/blender --python --background bpy_render_views.py -- --json_path data_objaverse/obj_name_path_0_50.json --output_path data_objaverse_render/ --num_views 100 --resolution 800 800 --device cuda


# convert into NeRFStudio data format
python transform_ns_format.py --input_dir data_objaverse_render/
```

### Run nerfstudio model training & inference
```
# example: training object "1ec7f"
ns-train nerfacto --pipeline.model.background-color random --pipeline.model.disable-scene-contraction True --pipeline.model.proposal-initial-sampler uniform --pipeline.model.near-plane 0.4 --pipeline.model.far-plane 6. --pipeline.datamanager.camera-optimizer.mode off --pipeline.model.use-average-appearance-embedding False --pipeline.model.distortion-loss-mult 0 --data data_objaverse_render/1ec7f/

# visualize in viewer
ns-viewer --load-config outputs/1ec7f/nerfacto/2023-11-13_234331/config.yml

# render videos from interpolated training path
ns-render interpolate --load-config outputs/1ec7f/nerfacto/2023-11-13_234331/config.yml --output-path outputs/1ec7f/render.mp4
```