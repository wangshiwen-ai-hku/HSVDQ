cd /homedata/swwang/projects/HSVDQ

conda create -n hsvdq python=3.10 -y
conda activate hsvdq

pip install -U pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-hsvdquant.txt