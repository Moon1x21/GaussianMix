apt-get update
apt-get install -y libgl1-mesa-glx
apt-get install -y libglib2.0-0

pip install opencv-contrib-python

HERE=$(dirname $(realpath $0))
cd $HERE
python main.py --dataset cifar100 --model WideResNetDropout --depth 28\
 --params 10 --beta_of_ricap 0.3 --stype 'mid' --postfix ricap0.3 
