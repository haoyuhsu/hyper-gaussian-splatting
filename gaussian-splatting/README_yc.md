

# Gaussian Splatting

## building blender
Reference: https://wiki.blender.org/wiki/Building_Blender/Linux/Ubuntu
sudo apt install build-essential git subversion cmake libx11-dev libxxf86vm-dev libxcursor-dev libxi-dev libxrandr-dev libxinerama-dev libegl-dev
sudo apt install libwayland-dev wayland-protocols libxkbcommon-dev libdbus-1-dev linux-libc-dev
sudo apt install libsm6



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
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

pip install tqdm plyfile
```


## Visualize


1. visualize during training

./install/bin/SIBR_remoteGaussian_app

2. visualize after training

cd SIBR_viwers
# ./install/bin/SIBR_PointBased_app --path ../data/tandt_db/tandt/train --outPath ../output/04f3aea1-8
./install/bin/SIBR_gaussianViewer_app --iteration 30000 --m ../output/04f3aea1-8 --path ../data/tandt_db/tandt/train