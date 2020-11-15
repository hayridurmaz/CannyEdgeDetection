from statistics import mean

import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy import ndimage
from skimage import img_as_ubyte

weak = np.int32(75)
strong = np.int32(255)
lowThresholdRatio = 0.05
highThresholdRatio = 0.25


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

    return newImage[:, :, 0]


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
    grayImage = np.array(img)
    grayImage.setflags(write=1)
    # for i in range(3):
    #     grayImage[:, :, i] = Avg
    # return grayImage

    grayImage[:, :, 0] = Avg
    return grayImage[:, :, 0]


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
    im_filtered[:, :] = convolution(image[:, :], gaussian_filter)
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

    '''
    note: when x and y filtered matrices are same, it overrides.
    '''
    image = image.astype('int32')  # int8 calculations not enough
    im_filtered_x = np.zeros_like(image)
    im_filtered_y = np.zeros_like(image)
    im_filtered_x[:, :] = convolution(image[:, :], sobel_kernel_x)
    im_filtered_y[:, :] = convolution(image[:, :], sobel_kernel_y)

    G = np.hypot(im_filtered_x, im_filtered_y)
    G = G / G.max() * 255
    theta = np.arctan2(im_filtered_y, im_filtered_x)
    return (G.astype(np.uint8), theta)


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res


def hysteresis(img):
    M, N = img.shape

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i, j] == weak):
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    return img


def BlurImage(image):
    mean_kernel = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    ) / 9.0
    im_filtered = np.zeros_like(image)
    im_filtered[:, :] = convolution(image[:, :], mean_kernel)
    return im_filtered


def showPlot(img):
    plt.imshow(img)
    plt.show()


def cannyEdgeDetection(img):
    img = img_as_ubyte(img)
    # img = cv2.imread('deneme.png')
    showPlot(img)
    if len(img.shape) == 3:
        img = rgb_to_gray(img)
    plt.set_cmap(plt.get_cmap(name='gray'))
    showPlot(img)
    img = BlurImage(img)
    # img = GaussianBlurImage(img, 2)
    imgplot = plt.imshow(img)
    plt.show()
    img, D = FindGradients(img)
    showPlot(img)
    img = non_max_suppression(img, D)
    showPlot(img)
    img = threshold(img)
    showPlot(img)
    img = hysteresis(img)
    showPlot(img)


def readImage(path='images/Lenna.png'):
    # img = cv2.imread('images/Lenna.png')
    img = mpimg.imread(path)
    # img = mpimg.imread('deneme.png')
    return img


if __name__ == '__main__':
    input = open("input.txt", "r")
    input = input.readlines()
    for i in range(len(input)):
        print(input[i])
        if i == 0:
            inputNumbers = input[i].split(" ")
            weak = np.int32(int(inputNumbers[0]))
            strong = np.int32(int(inputNumbers[1]))
            lowThresholdRatio = float(inputNumbers[2])
            highThresholdRatio = float(inputNumbers[3])
        else:
            img = readImage(input[i][:-1])
            cannyEdgeDetection(img)
