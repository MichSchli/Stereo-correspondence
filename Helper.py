__author__ = 'Michael'

#Define a convolution formula:
def convolute(x,y, img, kernel):
    #First, calculate width of kernel:
    kernel_width = len(kernel)
    im_h = img.shape[0]-1
    im_w = img.shape[1]-1

    #Define an accumulator:
    acc=0

    #Iterate through the kernel:
    itera = [elem - kernel_width/2 for elem in range(0, kernel_width)]
    for i in itera:
        for j in itera:
            img_x = x+i
            img_y = y+j
            if not (img_x<0 or img_x > im_h or img_y<0 or img_y > im_w):
                acc+=img[img_x, img_y]*kernel[i+kernel_width/2, j+kernel_width/2]

    #Return the accumulated result:
    return acc


#This is a direct translation of the interp-function found on absalon into python:
def interp(startim, point, values, maxitt, l):

    itt  = 0
    dimx = len(startim)-1
    dimy = len(startim[0])-1
    Ipim = startim
    temp = startim
    while itt < maxitt:
        for x in range(1,dimx):
            for y in range(1, dimy):
                av = Ipim[x-1,y] + Ipim[x+1,y] + Ipim[x,y-1] + Ipim[x,y+1]
                if point[x,y] == 1:
                    temp[x,y] = (values[x,y] + l*av)/(1.0 + 4*l)
                else:
                    temp[x,y] = av/4.0

        # copy values to borders
        for y in range(0, dimy+1):
            temp[0,y] = temp[2,y]
            temp[dimx,y] = temp[dimx-2,y]

        for x in range(0,dimy+1):
            temp[x,0] = temp[x,2]
            temp[x,dimy] = temp[x,dimy-2]

        Ipim = temp
        itt += 1

    return Ipim

import math
import numpy as np
from skimage.viewer import ImageViewer

#Direct translation of testinterp:
def testinterp(noiselevel, pointfrac):

    # Definitions
    dimx     = 100
    dimy     = 100
    dens     = 0.1
    dx2      = dimx/2
    dy2      = dimy/2
    startval = 3.0
    l   = 0.01
    maxitt   = 500

    # define ground truth
    GT = np.zeros((dimx, dimy))
    GT[0:dx2,0:dy2] = - 10.0
    GT[0:dx2,dy2+1:] = 0.0
    GT[dx2+1:,0:dy2] = 5.0
    GT[dx2+1:,dy2+1:] = 15.0

    print np.random.randn(dimx,dimy).shape
    print GT.shape

    # generate data
    Data = GT + noiselevel*np.random.randn(dimx,dimy)
    points = (np.random.rand(dimx,dimy) < pointfrac)
    values = np.multiply((1*points), Data)
    start = startval*np.ones((dimx,dimy))

    print "Interpret!"
    Ipim = interp(start, points, values, maxitt, l)
    print "Done!"

    # compare and display result

    err1 = (Data - Ipim)
    err = np.absolute(Data - Ipim)
    errsq = err1.dot(err1)

    rms= math.sqrt(errsq.sum())/float(dimx*dimy)
    print 'average reconstruction error:', rms

    shGT = GT - GT.min()
    shGT = shGT/float(shGT.max())
    shIp = Ipim - Ipim.min()
    shIp = shIp/float(shIp.max())
    sher = err - err.min()
    sher = sher/float(sher.max())

    #h = figure(1);
    #subplot(2,2,1);

    print "ground truth"
    view = ImageViewer(shGT)
    view.show()

    print 'Reconstruction'
    view = ImageViewer(shIp)
    view.show()


    print 'Sparse point position'
    view = ImageViewer(points)
    view.show()

    print 'Reconstruction error'
    view = ImageViewer(sher)
    view.show()


if __name__ == '__main__':
    testinterp(0.1, 0.1)