export DATA_ROOT_DIR=./data_objaverse/
export START_INDEX=0
export END_INDEX=-1  # -1: whole list
export NUM_VIEWS=100  # number of views sampled per object asset

mkdir -p data_objaverse/
mkdir -p data_objaverse_render/

# download assets from Objaverse
python download_objaverse.py \
    --output_dir $DATA_ROOT_DIR \
    --start_index $START_INDEX \
    --end_index $END_INDEX \
    --use_lvis  # if need specific categories