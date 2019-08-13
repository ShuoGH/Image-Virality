OMP_NUM_THREADS=1
export OMP_NUM_THREADS

# python train.py --device cpu --epochs 10 --check_point 0
python train.py --device cuda:0 --epochs 10 --check_point 0