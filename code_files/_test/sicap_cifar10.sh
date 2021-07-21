apt-get update
apt-get install -y libgl1-mesa-glx
apt-get install -y libglib2.0-0

pip install opencv-contrib-python

HERE=$(dirname $(realpath $0))
script_dir=$(dirname $0)
echo $HERE
echo $script_dir
cd $HERE
python main.py --dataset cifar10 --model ResNet18 --depth 28\
 --params 10 --beta_of_ricap 0.3 --postfix 'P:test' --stype 'mid'

 