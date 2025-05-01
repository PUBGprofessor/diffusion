import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.utils import make_grid
import cv2
import einops
import numpy as np
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

def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5 # [-1, 1]到[0, 1]

def show_images(images, save_path=None, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    # ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    grid = make_grid(denorm(images.detach()[:nmax]), nrow=8)
    ax.imshow(grid.permute(1, 2, 0).cpu())  # .cpu() 保证能显示/保存到本地
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

def sample_imgs(ddpm,
                net,
                output_path,
                n_sample=81,
                device='cuda',
                simple_var=True):
    shape = (n_sample, 3, 64, 64)  # 1, 3, 28, 28
    imgs = ddpm.sample_backward(shape,
                                net,
                                device=device,
                                simple_var=simple_var).detach().cpu()
    imgs = (imgs + 1) / 2 * 255
    imgs = imgs.clamp(0, 255)
    imgs = einops.rearrange(imgs,
                            '(b1 b2) c h w -> (b1 h) (b2 w) c',
                            b1=int(n_sample**0.5))

    imgs = imgs.numpy().astype(np.uint8)

    cv2.imwrite(output_path, imgs)