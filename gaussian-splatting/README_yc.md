

# Gaussian Splatting

## Setup environment

1. install cuda=11.8
```
# from this page: https://developer.nvidia.com/cuda-11-8-0-download-archive
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run # --tmpdir=/home/yc/tmp if /tmp is not enough
```

```
conda create -n gs python=3.8
conda activate gs

# Install packages
# conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # check your cuda
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia # aws

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

pip install tqdm plyfile

```

```
## with pytorch3d
conda create --name gs-v2 python=3.9 -y && conda activate gs-v2

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y # aws with cuda=11.8

# install pytorch3d. remember to export path of cuda for 11.8
# export PATH="/usr/local/cuda-11.8/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

pip install tqdm plyfile einops

```


## Training
1. train one gs
```
./train_gs.sh
```

2. train batch
```
python get_dist_train_cmd.py
```
which will output `train_gs_cmd_{today}.txt`
Run command written in `train_gs_cmd_{today}.txt` on server
```
# in tmux
cd /nfs/ycheng/hyper-gaussian-splatting/gaussian-splatting; sa gs
```


## Visualize


1. visualize during training

./install/bin/SIBR_remoteGaussian_app

2. visualize after training

cd SIBR_viwers
# ./install/bin/SIBR_PointBased_app --path ../data/tandt_db/tandt/train --outPath ../output/04f3aea1-8
./install/bin/SIBR_gaussianViewer_app --iteration 30000 --m ../output/04f3aea1-8 --path ../data/tandt_db/tandt/train

3. rendering
./rend_gs.sh