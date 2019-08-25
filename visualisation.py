import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import PIL.Image as Image


'''
Do the visulization work of the images after training.
'''


def show_img(img_tensor):
    PIL_img = transforms.functional.to_pil_image(img_tensor)
    # PIL_img.show()
    plt.imshow(PIL_img)
    plt.show()


def show_img_in_notebook(img_tensor):
    '''
    The tensor after transformation is not the same shape with PIL image
    '''
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


def theta_to_abs_coordinate(theta, image_size):
    '''
    args:
        theta: parameters of the affine transformation a,t_x,t_y. s is the scale value, t is the translation value
        image_size: (C * H * W)
    return:
        coordinates (x1,y1) and the size of H, W
    '''
    s, tx, ty = theta
    s = abs(s)  # when the s is below 0, it means rotation
    x_all = image_size[2]
    y_all = image_size[1]

    size_part_h = s * image_size[1]
    size_part_w = s * image_size[2]

    x1 = round((1 - s + tx) * x_all / 2)
    y1 = round((1 - s + ty) * y_all / 2)
    return (int(x1), int(y1)), int(size_part_h), int(size_part_w)


def plot_img_after_affine(x1, y1, H, W, image):
    '''
    args:
        x1,y1: the coordinates of the x1,y1 
        H,W: the size of the local part
        image: the image tensor (C * H * W)
    '''
    img_part = image[:, y1: y1 + H, x1: x1 + W].permute(1, 2, 0)
    plt.imshow(img_part)
    plt.show()


def localise_part_visualisation(image, x1, y1, patch_size, color='r'):
    '''
    args:
        patch_size: tuple of (H,W)
    '''
    image = image.permute(1, 2, 0)
    H, W = patch_size
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)
    # Create a Rectangle patch
    rect = patches.Rectangle((x1, y1), W, H, linewidth=1,
                             edgecolor='r', facecolor='none', lw=2)
    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()
