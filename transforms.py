import torch
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pandas as pd
import torchvision
from torchvision import transforms
from PIL import Image


'''
Transforms which could be used on the images beforfe feeding into the data loader

The info of pictures data:
    - tensor of image: 3*360*360 (C H W)
Note:
    - Size of input: (height,width,channels)  the return of cv2.imread()
    - the input of the torch model should be (channels,height,width) (so I'd better use torchvision.transform)
    - most transforms function within torch is based on the PIL data type. Therefore, I would prefer PIL.
'''


def transforms_img():
    '''
    Just basic transform of the images.
    input:
        PIL image data type
    return:
        PIL to tensor
    '''
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    return preprocess


if __name__ == '__main__':
    IMG_PATH_TRAIN = "./images"
    image_list = np.array(os.listdir(IMG_PATH_TRAIN))

    for i in range(4):
        index = np.random.randint(0, 100)
        img_name = image_list[index]
        print(img_name)
        im = Image.open(os.path.join(IMG_PATH_TRAIN, img_name))
        im.show()
        print(type(im))
        print(transforms.functional.to_tensor(im).size())

        # apply the transform
        # show the image after the transformation
        transform = transforms_img()
        output = transform(im)
        im_out = transforms.functional.to_pil_image(output)
        im_out.show()
        print(output.size())
