OMP_NUM_THREADS=1
export OMP_NUM_THREADS

# python train.py --device cpu --epochs 10 --check_point 0
python train.py --device cuda:0 --freeze_pretrained 1 --pair_mode 4 --epochs 50 --check_point 0
