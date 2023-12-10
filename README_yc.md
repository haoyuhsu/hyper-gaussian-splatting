

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
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tqdm plyfile
```


## Visualize
cd SIBR_viwers
# ./install/bin/SIBR_PointBased_app --path ../data/tandt_db/tandt/train --outPath ../output/04f3aea1-8
./install/bin/SIBR_gaussianViewer_app --iteration 30000 --m ../output/04f3aea1-8 --path ../data/tandt_db/tandt/train