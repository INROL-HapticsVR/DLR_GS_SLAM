# GS-ICP SLAM Installation and Execution Guide

## Prerequisites
- **Operating System**: Ubuntu 20.04
- **Package Manager**: Conda
- **Python Version**: 3.9

## Setting Up the Environment

### 1. Create and Activate Conda Environment
```bash
conda create -n gsicpslam python==3.9 -y
conda activate gsicpslam
```

### 2. Install PyTorch and CUDA Dependencies
```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Submodules
```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

### 5. Build and Install `fast_gicp`
```bash
cd submodules/fast_gicp
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..
python setup.py install --user
cd ../..
```

## Running GS-ICP SLAM
```bash
python -W ignore gs_icp_slam.py --rerun_viewer
```

