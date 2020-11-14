from statistics import mean

import cv2
import numpy as np
from matplotlib import pyplot as plt


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


def convolution(image, kernel):
    """
    This function which takes an image and a kernel and returns the convolution of them.
    :param image: a numpy array of size [image_height, image_width].
    :param kernel: a numpy array of size [kernel_height, kernel_width].
    :return: a numpy array of size [image_height, image_width] (convolution output).
    """
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))
    # convolution output filled with zeros
    output = np.zeros_like(image)

    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image

    # Loop over every pixel of the image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # element-wise multiplication of the kernel and the image
            output[x, y] = (kernel * image_padded[x: x + 3, y: y + 3]).sum()

    return output


def GaussianBlurImage(image, sigma):
    filter_size = 3
    filter_size = int(filter_size) // 2
    x, y = np.mgrid[-filter_size:filter_size + 1, -filter_size:filter_size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    gaussian_filter = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    im_filtered = np.zeros_like(image)
    for c in range(3):
        im_filtered[:, :, c] = convolution(image[:, :, c], gaussian_filter)
    return im_filtered


def FindGradients(image):
    sobel_kernel_x = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]
    ) / 8.0
    sobel_kernel_y = np.array(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]]
    ) / 8.0

    im_filtered = np.zeros_like(image)
    for c in range(3):
        im_filtered[:, :, c] = convolution(image[:, :, c], sobel_kernel_x)
    for c in range(3):
        im_filtered[:, :, c] = convolution(image[:, :, c], sobel_kernel_y)
    return im_filtered


def BlurImage(image):
    mean_kernel = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    ) / 9.0
    im_filtered = np.zeros_like(image)
    for c in range(3):
        im_filtered[:, :, c] = convolution(image[:, :, c], mean_kernel)
    return im_filtered


if __name__ == '__main__':
    img = cv2.imread('images/Lenna.png')
    img = rgb_to_gray(img)
    # img = BlurImage(img)
    img = GaussianBlurImage(img, 1)
    img = FindGradients(img)
    imgplot = plt.imshow(img)
    plt.show()
