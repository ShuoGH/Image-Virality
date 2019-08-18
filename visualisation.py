import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
'''
Do the visulization work of the images after training.
'''


def show_img(img_tensor):
    PIL_img = transforms.functional.to_pil_image(img_tensor)
    PIL_img.show()


def show_img_in_notebook(img_tensor):
    plt.imshow(img_tensor.permute(1, 2, 0))
    plt.show()


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of tensors C H W

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.size()[0] == 2:
            plt.gray()
        plt.imshow(image.permute(1, 2, 0))
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
