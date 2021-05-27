import numpy as np
from scipy.ndimage import filters
from scipy.ndimage import shift
from skimage import io
import matplotlib.pyplot as plt
import argparse

EPS = np.finfo(np.float32).eps

def command_line_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('input_image', type=str)
    parser.add_argument('kernel', type=str)
    parser.add_argument('output_image', type=str)
    parser.add_argument('noise_level', type=float)
    return parser


def create_blurred_noisy(image, kernel, noise_level):
    kernel_sum = np.sum(kernel)
    kernel = kernel / kernel_sum
    bad_image = filters.convolve(image, kernel) + np.random.normal(0, noise_level, image.shape)
    return bad_image


def deblur(blurred, kernel, noise_level, alpha=0.1, beta=0.1, mu=0.1):
    kernel_sum = np.sum(kernel)
    kernel = kernel / kernel_sum
    z = blurred
    v = np.zeros(blurred.shape)
    conv_kernel_t_u = 2 * filters.convolve(blurred, kernel[::-1, ::-1])

    for i in range(100):
        norm, grad_tv = tv_norm(z)
        grad = grad_conv(z, kernel, conv_kernel_t_u) + alpha * grad_tv
        v = mu * v - grad
        z = z + beta * v
        z = np.clip(z, 0, 255)

    z = z.astype('uint8')
    return z

def grad_conv(z, kernel, conv_kernel_t_u):
    return 2 * filters.convolve(filters.convolve(z, kernel), kernel[::-1, ::-1]) - conv_kernel_t_u

def tv_norm(z):
    x_diff = z - np.roll(z, -1, axis=1)
    y_diff = z - np.roll(z, -1, axis=0)
    grad_norm2 = x_diff**2 + y_diff**2 + EPS
    norm = np.sum(np.sqrt(grad_norm2))
    dgrad_norm = 0.5 / np.sqrt(grad_norm2)
    dx_diff = 2 * x_diff * dgrad_norm
    dy_diff = 2 * y_diff * dgrad_norm
    grad = dx_diff + dy_diff
    grad[:, 1:] -= dx_diff[:, :-1]
    grad[1:, :] -= dy_diff[:-1, :]
    return norm, grad


parser = command_line_parser()
args = parser.parse_args()

image = io.imread(args.input_image)
image = image.astype('float64')
if (len(image.shape) > 2):
    image = image[:,:,0]
kernel = io.imread(args.kernel)
kernel = kernel.astype('float64')
if (len(kernel.shape) > 2):
    kernel = kernel[:,:,0]

res_image = deblur(image, kernel, args.noise_level, alpha=args.noise_level / 5)
io.imsave(args.output_image, res_image)
#io.imshow(res_image)
#plt.show()

#my_blurred = create_blurred_noisy(image, kernel, args.noise_level)
#my_blurred = np.clip(my_blurred, 0, 255)
#my_blurred = my_blurred.astype('uint8')
#io.imsave(args.output_image, my_blurred)
#io.imshow(my_blurred)
#plt.show()

