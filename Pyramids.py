__author__ = 'Michael'

import numpy as np
from skimage.filter import gaussian_filter
from skimage.io import imread
from skimage.viewer import ImageViewer
from skimage import color

'''
Filtering:
'''

#Define a function to blur an image:
def blur(img, sigma=1.41):
    return gaussian_filter(img, sigma)

'''
Sampling:
'''

#Define a function to upsample and smoothe an image:
def upsample(image, sample_rate=2):
    #Create a canvas of zeroes corresponding to the desired image size:
    dim = image.shape
    canvas = np.zeros(((dim[0]-1)*sample_rate+1, (dim[1]-1)*sample_rate+1))

    #Paint the image onto the canvas:
    canvas[::sample_rate, ::sample_rate] = image

    #Blur the image and return the result:
    return blur(canvas)

#Define a function to smoothe and downsample an image
def downsample(image, sample_rate=2):
    #Blur the image:
    blurred_image = blur(image)

    #Take samples:
    return blurred_image[::sample_rate, ::sample_rate]

'''
Pyramid construction:
'''

#Define a function to create a downsampling pyramid:
def down_pyramid(image, levels=4, sample_rate=2):
    #Ensure that downsampling is actually possible:
    assert image.shape[0] > sample_rate**levels
    assert image.shape[1] > sample_rate**levels

    #Initialize the pyramid:
    pyramid = [image]

    #Iteratively construct the pyramid:
    for _ in xrange(levels-1):
        image = downsample(image, sample_rate=sample_rate)
        pyramid.append(image)

    return pyramid

#Define a function to create an upsampling pyramid:
def up_pyramid(image, levels=4, sample_rate=2):
    #Initialize the pyramid:
    pyramid = [image]

    #Iteratively construct the pyramid:
    for _ in xrange(levels-1):
        image = upsample(image, sample_rate=sample_rate)
        pyramid.append(image)

    return pyramid

'''
Testing playground:
'''

if __name__ == '__main__':
    img = imread('pentagonL.jpg')
    img2 = color.rgb2gray(img)

    pyram = down_pyramid(img2)

    for image in pyram:
        viewer = ImageViewer(image)
        viewer.show()

    pyram2 = up_pyramid(pyram[-1])

    for image in pyram2:
        viewer = ImageViewer(image)
        viewer.show()
