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


def make_pyramid(image, levels=4, sample_rate=2):
    pass

def upsample(image, sample_rate=2):
    pass

'''
Testing playground:
'''

if __name__ == '__main__':
    pass