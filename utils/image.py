img_size = None

def set_img_size(imgsize):
    global img_size
    img_size = imgsize

def get_img_shape():
    """
    Get the shape of the image.
    :return: (C, H, W)
    """
    if img_size is None:
        raise TypeError("img_size not set!")
    return img_size
    # return 1, 28, 28

