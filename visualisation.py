from torchvision import transforms
'''
Do the visulization work of the images after training.
'''


def show_img(img_tensor):
    PIL_img = transforms.functional.to_pil_image(img_tensor)
    PIL_img.show()
    return
