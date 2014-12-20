__author__ = 'Michael'

import numpy as np
import math

'''
Utility functions:
'''

#Define a Gaussian Blur:
def gaussian_blur(w, sigma):
    m = np.zeros((2*w+1,2*w+1))

    for i in range(-w,w+1):
        for j in range(-w,w+1):
            m[i,j] = 1/(2.0*math.pi*(sigma**2))*math.exp(-((i-w-1)**2 + (j-w-1)**2)/(2.0*(sigma**2)))

    return m/float(m.sum())

'''
Filtering:
'''

def blur(img):
    pass


'''
Sampling:
'''

#Define a function to upsample and smoothe an image:
def upsample(image, sample_rate=2):
    pass

#Define a function to smoothe and downsample an image
def downsample(image, sample_rate=2):
    #Blur the image:
    blurred_image = blur(image)

    #Take samples:
    return blurred_image[::sample_rate, ::sample_rate]

'''
Pyramid construction:
'''

def make_pyramid(image, levels=4, sample_rate=2):
    pass

'''
Testing playground:
'''

if __name__ == '__main__':
    x = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print x

    print x[::4, ::4].shape

