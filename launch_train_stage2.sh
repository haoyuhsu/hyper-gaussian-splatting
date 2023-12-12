
# today=$(date +%Y%m%d_%H%M%S)
DATE_WITH_TIME=`date "+%Y-%m-%dT%H"`

gpu_id=$1
kl_weight=$2
# kl_weight=0

# lr=1e-3
lr=$3

opac_weight=1e-3
scale_weight=1e-2
z_dim=2048

# TODO: norm gt rots?

exp_name="${DATE_WITH_TIME}-lr_${lr}-z_${z_dim}-kl_${kl_weight}-opac_${opac_weight}-scale_${scale_weight}"

cmd="python train_second_stage.py --exp_name ${exp_name} --lr ${lr} --kl_weight ${kl_weight} \
                    --opac_weight ${opac_weight} --scale_weight ${scale_weight}"
            
echo CUDA_VISIBLE_DEVICES=${gpu_id} $cmd
CUDA_VISIBLE_DEVICES=${gpu_id} $cmd


##
# ./launch_train_stage2.sh 1 0
# ./launch_train_stage2.sh 2 1e-2
# ./launch_train_stage2.sh 3 1e-3
# ./launch_train_stage2.sh 4 1e-4
# ./launch_train_stage2.sh 5 1e-5
# ./launch_train_stage2.sh 6 10
# ./launch_train_stage2.sh 7 100