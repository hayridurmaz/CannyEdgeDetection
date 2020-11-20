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
highThresholdRatio = 20


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
    grayImage[:, :, 0] = Avg
    return grayImage[:, :, 0]


def convolution(image, kernel):
    """
    This function which takes an image and a kernel and returns the convolution of them.
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


def applyNonMaximaSuppression(img, D):
    rowSize, columnSize = img.shape
    Z = np.zeros((rowSize, columnSize), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, rowSize - 1):
        for j in range(1, columnSize - 1):
            # try:
            q = 255
            r = 255
            currentAngle = D[i][j]
            currentAngle = currentAngle * 180. / np.pi
            if currentAngle < 0:
                currentAngle += 180

            # angle 0
            if (0 <= currentAngle < 22.5) or (157.5 <= currentAngle <= 180):
                q = img[i, j + 1]
                r = img[i, j - 1]
            # angle 45
            elif 22.5 <= currentAngle < 67.5:
                q = img[i + 1, j - 1]
                r = img[i - 1, j + 1]
            # angle 90
            elif 67.5 <= currentAngle < 112.5:
                q = img[i + 1, j]
                r = img[i - 1, j]
            # angle 135
            elif 112.5 <= currentAngle < 157.5:
                q = img[i - 1, j - 1]
                r = img[i + 1, j + 1]

            if (img[i, j] >= q) and (img[i, j] >= r):
                Z[i, j] = img[i, j]
            else:
                Z[i, j] = 0

        # except IndexError as e:
        #     pass

    return Z


def threshold(img):
    rowSize, columnSize = img.shape
    thresholdedMap = np.zeros((rowSize, columnSize))

    for row in range(1, rowSize - 1):
        for col in range(1, columnSize - 1):

            # If pixel value is higher than highthreshold, it is strong edge
            if img[row, col] >= highThresholdRatio:
                thresholdedMap[row, col] = strong

            # If pixel value is in between thresholds, it is weak edge
            elif lowThresholdRatio <= img[row, col] and img[row, col] < highThresholdRatio:
                thresholdedMap[row, col] = weak

            # If pixel value is lower than lowthreshold, it is non-relevant pixel, zero out it
            elif img[row, col] < lowThresholdRatio:
                thresholdedMap[row, col] = 0

    return thresholdedMap


def applyHysteresisThreshold(img):
    # Get size of image
    rowSize = img.shape[0]
    columnSize = img.shape[1]

    finalImage = np.zeros((rowSize, columnSize))

    # Loop over thresholded map to find weak edges which indeed is strong edge
    for row in range(1, rowSize - 1):
        for col in range(1, columnSize - 1):

            if img[row, col] == weak:

                # Look at 8 neigbours of current pixel to find and connected strong value
                if ((img[row + 1, col - 1] == strong) or (img[row + 1, col] == strong) or (
                        img[row + 1, col + 1] == strong)
                        or (img[row, col - 1] == strong) or (img[row, col + 1] == strong)
                        or (img[row - 1, col - 1] == strong) or (img[row - 1, col] == strong) or (
                                img[row - 1, col + 1] == strong)):
                    finalImage[row, col] = 1
                else:
                    finalImage[row, col] = 0

            elif img[row, col] == strong:
                finalImage[row, col] = 1

    return finalImage


def BlurImage(image):
    '''
    This function blurs image with an average filter
    :param image:
    :return:
    '''
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


def saveOutput(img, name):
    mpimg.imsave("output/" + name, img)


def cannyEdgeDetection(img, name):
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
    img = applyNonMaximaSuppression(img, D)
    showPlot(img)
    img = threshold(img)
    showPlot(img)
    img = applyHysteresisThreshold(img)
    showPlot(img)
    saveOutput(img, name)


def readImage(path='images/Lenna.png'):
    # img = cv2.imread('images/Lenna.png')
    img = mpimg.imread(path)
    # img = mpimg.imread('deneme.png')
    return img


if __name__ == '__main__':
    inputImages = open("input.txt", "r")
    inputImages = inputImages.readlines()
    for i in range(len(inputImages)):
        print(inputImages[i])
        if i == 0:
            inputNumbers = inputImages[i].split(" ")
            weak = np.int32(int(inputNumbers[0]))
            strong = np.int32(int(inputNumbers[1]))
            lowThresholdRatio = float(inputNumbers[2])
            highThresholdRatio = float(inputNumbers[3])
        else:
            img = readImage(inputImages[i][:-1])
            cannyEdgeDetection(img, inputImages[i][:-1])
