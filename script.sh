OMP_NUM_THREADS=1
export OMP_NUM_THREADS

# 1. For classifier training
# python train_classifier.py --batch_size 64 --device cuda:0 --model alexnet --balance_data 1 --freeze_features 1 --epochs 50 --check_point 0 

# python train_classifier.py --batch_size 64 --device cuda:0 --model alexnet --freeze_features 0 --epochs 50 --check_point 0 

# python train_classifier.py --batch_size 64 --device cuda:0 --model vgg --balance_data 1 --freeze_features 1 --epochs 50 --check_point 0 

# python train_classifier.py --batch_size 64 --device cuda:0 --model vgg --freeze_features 0 --epochs 50 --check_point 0 

# python train_classifier.py --batch_size 64 --device cuda:0 --model resnet --balance_data 1 --freeze_features 1 --epochs 50 --check_point 0 

# python train_classifier.py --batch_size 64 --device cuda:0 --model resnet --freeze_features 0 --epochs 50 --check_point 0 

# python train_classifier.py --batch_size 64 --device cuda:0 --model densenet --balance_data 1 --freeze_features 1 --epochs 50 --check_point 0 

# python train_classifier.py --batch_size 64 --device cuda:0 --model densenet --freeze_features 0 --epochs 50 --check_point 0 

# 2. For siamese network training
python train_siamese.py --device cuda:0 --freeze_locnet 0 --freeze_ranknet 0 --pair_mode 4 --epochs 50 --check_point 0

# python train_siamese.py --device cuda:0 --freeze_locnet 1 --freeze_ranknet 1 --pair_mode 4 --epochs 50 --check_point 0

# python train_siamese.py --device cuda:0 --freeze_locnet 1 --freeze_ranknet 0 --pair_mode 4 --epochs 50 --check_point 0

# python train_siamese.py --device cuda:0 --freeze_locnet 0 --freeze_ranknet 1 --pair_mode 4 --epochs 50 --check_point 0