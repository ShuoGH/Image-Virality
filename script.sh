OMP_NUM_THREADS=1
export OMP_NUM_THREADS

# python train.py --device cpu --epochs 10 --check_point 0
# python train_siamese.py --device cuda:0 --freeze_pretrained 1 --learning_rate 0.001 --pair_mode 4 --epochs 50 --check_point 0

python train_classifier.py --batch_size 64 --device cuda:0 --model alexnet --freeze_features 1 --epochs 50 --check_point 0 

# python train_classifier.py --batch_size 64 --device cuda:0 --model alexnet --freeze_features 0 --epochs 50 --check_point 0 

# python train_classifier.py --batch_size 64 --device cuda:0 --model vgg --freeze_features 1 --epochs 50 --check_point 0 

# python train_classifier.py --batch_size 64 --device cuda:0 --model vgg --freeze_features 0 --epochs 50 --check_point 0 

# python train_classifier.py --batch_size 64 --device cuda:0 --model resnet --freeze_features 1 --epochs 50 --check_point 0 

# python train_classifier.py --batch_size 64 --device cuda:0 --model resnet --freeze_features 0 --epochs 50 --check_point 0 

# python train_classifier.py --batch_size 64 --device cuda:0 --model densenet --freeze_features 1 --epochs 50 --check_point 0 

# python train_classifier.py --batch_size 64 --device cuda:0 --model densenet --freeze_features 0 --epochs 50 --check_point 0 