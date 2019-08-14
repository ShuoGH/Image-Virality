import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
'''
Do the visulization work of the images after training.
'''


def show_img(img_tensor):
    PIL_img = transforms.functional.to_pil_image(img_tensor)
    PIL_img.show()


def show_img_in_notebook(img_tensor):
    plt.imshow(img_tensor.permute(1, 2, 0))
    plt.show()
