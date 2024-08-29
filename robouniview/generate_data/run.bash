export PATH=$PATH:/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mlm/liufanfan/workspace/RoboFlamingo
export PYTHONPATH=$PYTHONPATH:/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mlm/liufanfan/workspace/RoboFlamingo

source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mlm/yanfeng/software/anaconda3/bin/activate RoboFlamingo_yiyang

#bash install_da.sh
sudo apt-get -y install libegl1-mesa libegl1
sudo apt-get -y install libgl1

sudo apt-get update -y -qq
sudo apt-get install -y -qq libegl1-mesa libegl1-mesa-dev

sudo apt install -y mesa-utils libosmesa6-dev llvm
sudo apt-get -y install meson
sudo apt-get -y build-dep mesa

sudo apt-get -y install freeglut3
sudo apt-get -y install freeglut3-dev

sudo apt update -y 
sudo apt install -y xvfb


sudo apt-get install -y --reinstall libgl1-mesa-dri


Xvfb :99 -screen 0 1024x768x16 &

export DISPLAY=:99
#export PYTHONPATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mlm/liufanfan/workspace/RoboFlamingo
export EVALUTION_ROOT=$(pwd)

#python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=2  --master_port=6042 robouniview/train/train_calvin.py \ --use_aug  \
#python3 robouniview/new_data/generate_data.py 
python3 robouniview/generate_data/generate_data.py 
   
