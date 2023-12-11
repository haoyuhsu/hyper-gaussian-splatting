
# Usage: bash rend_objaverse.sh gpu_id start_ix end_ix
gpu_id=$1
start_index=$2
end_index=$3

blender_path="/nfs/ycheng/blender-3.5.1-linux-x64/blender"

cmd="$blender_path --background --python bpy_render_views.py -- --data_dir ./data_objaverse --output_path data_objaverse_render/ \
     --num_views 100 --resolution 800 800 --device cuda --start_index $start_index --end_index $end_index"
echo CUDA_VISIBLE_DEVICES=$gpu_id $cmd
CUDA_VISIBLE_DEVICES=$gpu_id $cmd
# CUDA_VISIBLE_DEVICES=$gpu_id $blender_path $cmd


# testing
# [*] total rend: 18 render start name: 7eb655072ffb4ca1b2de9b13c11f3eaa end name: 818d7994915d447bbf05be34a0b122fe

# cmd for tmux
# cd /nfs/ycheng/hyper-gaussian-splatting; sa gs