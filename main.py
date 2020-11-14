import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from statistics import mean


def grayscale(img):
    # second choice grayscale image
    # determining width and height of original image
    w, h = img.shape[:2]

    # new Image dimension with 4 attribute in each pixel
    newImage = np.zeros([w, h, 4])
    for i in range(w):
        for j in range(h):
            # ratio of RGB will be between 0 and 1

            lst = [float(img[i][j][0]), float(img[i][j][1]), float(img[i][j][2])]
            avg = float(mean(lst))
            newImage[i][j][0] = avg
            newImage[i][j][1] = avg
            newImage[i][j][2] = avg
            newImage[i][j][3] = 1  # alpha value to be 1

    return newImage


def rgb_to_gray(img):
    grayImage = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    # rates for grayscale image
    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    Avg = (R + G + B)
    grayImage = img

    for i in range(3):
        grayImage[:, :, i] = Avg

    return grayImage


def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output


# def zConv(m, K):
#     # input assumed to be numpy arrays Kr<=mrow, Kc<=mcol, Kernal odd
#     # edges wrap Top/Bottom, Left/Right
#     # Zero Pad m by kr,kc if no wrap desired
#     mc = m * 0
#     Kr, Kc = K.shape
#     kr = Kr // 2  # kernel center
#     kc = Kc // 2
#     for dr in range(-kr, kr + 1):
#         mr = np.roll(m, dr, axis=0)
#         for dc in range(-kc, kc + 1):
#             mrc = np.roll(mr, dc, axis=1)
#             mc = mc + K[dr + kr, dc + kc] * mrc
#     return mc

# def convolution(image,kernel):
#     image_h = image.shape[0]
#     image_w = image.shape[1]
#
#     kernel_h = kernel.shape[0]
#     kernel_w = kernel.shape[1]
#     for i in image_h:
#         for j in image_w:
#             acc=0
#             for k in kernel_h:
#                 for l in kernel_w:


# def convolution(oldimage, kernel):
#     # image = Image.fromarray(image, 'RGB')
#     image_h = oldimage.shape[0]
#     image_w = oldimage.shape[1]
#
#     kernel_h = kernel.shape[0]
#     kernel_w = kernel.shape[1]
#
#     if (len(oldimage.shape) == 3):
#         image_pad = np.pad(oldimage, pad_width=(
#             \(kernel_h // 2, kernel_h // 2), (kernel_w // 2,
#                                               \kernel_w // 2), (0, 0)), mode = 'constant',
#         \constant_values = 0).astype(np.float32)
#     elif (len(oldimage.shape) == 2):
#         image_pad = np.pad(oldimage, pad_width=(
#             \(kernel_h // 2, kernel_h // 2), (kernel_w // 2,
#                                               \kernel_w // 2)), mode = 'constant', constant_values = 0)
#         \.astype(np.float32)
#
#
#     h = kernel_h // 2
#     w = kernel_w // 2
#
#     image_conv = np.zeros(image_pad.shape)
#
#     for i in range(h, image_pad.shape[0] - h):
#         for j in range(w, image_pad.shape[1] - w):
#             # sum = 0
#             x = image_pad[i - h:i - h + kernel_h, j - w:j - w + kernel_w]
#             x = x.flatten() * kernel.flatten()
#             image_conv[i][j] = x.sum()
#     h_end = -h
#     w_end = -w
#
#     if (h == 0):
#         return image_conv[h:, w:w_end]
#     if (w == 0):
#         return image_conv[h:h_end, w:]
#     return image_conv[h:h_end, w:w_end]


def GaussianBlurImage(image, sigma):
    # image = imread(image)
    image = np.asarray(image)
    # print(image)
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    n = filter_size // 2

    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            x1 = 2 * np.pi * (sigma ** 2)
            x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            gaussian_filter[x + m, y + n] = (1 / x1) * x2

    im_filtered = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        im_filtered[:, :, c] = convolve2D(image[:, :, c], gaussian_filter)
    return (im_filtered.astype(np.uint8))


if __name__ == '__main__':
    img = mpimg.imread('Lenna.png')
    imgplot = plt.imshow(rgb_to_gray(img))
    plt.show()
