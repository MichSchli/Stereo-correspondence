__author__ = 'Michael'

'''
Imports:
'''
import Pyramids
import numpy as np
import Helper
from skimage.filter import hsobel, vsobel,canny
from skimage.io import imread
from skimage.viewer import ImageViewer
from skimage import color,img_as_ubyte
from skimage.draw import circle_perimeter,line
import math
import random

'''
Feature extraction:
'''

#Define a method to extract a gradient map from an image:
def gradient_map(image):
    h = hsobel(image)
    v = vsobel(image)
    return np.dstack((h,v))

#Define a method to extract a gradient orientation map from an image:
def gradient_orientation_map(image):
    h = hsobel(image)
    v = vsobel(image)
    return np.arctan2(h,v)

#Define a method to extract an edge map from an image:
def edge_map(image):
    return canny(image)

'''
Matching algorithm:
'''

#Define a wrapper method for the matching strategy, used for testing purposes:
def gradient_match_wrapper(image1, image2,search_radius = 40):

    left_edges = edge_map(image1)
    right_edges = edge_map(image2)
    left_gradients = gradient_map(image1)
    right_gradients = gradient_map(image2)

    #Pad the edge images:
    padded_left_edges = np.pad(left_edges, (search_radius,search_radius), 'constant', constant_values=(0,0))
    padded_right_edges = np.pad(right_edges, (search_radius,search_radius), 'constant', constant_values=(0,0))

    s =(left_edges.shape[0], left_edges.shape[1], 2)
    points = np.zeros_like(left_edges)
    values = np.empty(shape=s,dtype=int)

    #Iterate through the image:
    for x in xrange(search_radius, len(padded_left_edges)-search_radius):
        for y in xrange(search_radius, len(padded_left_edges[0])-search_radius):

            #Check if there is an edge:
            if padded_left_edges[x,y]:
                best_similarity = -1
                #Iterate through the window to see if there are other edges:
                for i in xrange(-search_radius, search_radius):
                    for j in xrange(-search_radius, search_radius):
                        if padded_right_edges[x+i,y+j]:
                            g1 = left_gradients[x-search_radius,y-search_radius]
                            g2 = right_gradients[x-search_radius+i,y-search_radius+j]

                            #Calculate cosine distance:
                            n1 = np.linalg.norm(g1)
                            n2 = np.linalg.norm(g2)

                            #We dont like zero gradients:
                            if n1 != 0 and n2 != 0:
                                #Compare it:
                                similarity = np.dot(g1, g2)/n1/n2

                                if similarity > best_similarity:
                                    points[x-search_radius,y-search_radius] = True
                                    values[x-search_radius,y-search_radius] = (x+i-search_radius,y+j-search_radius)
                                    best_similarity = similarity


    return points, values

#Define the matching strategy:
def match(left_edges, left_gradients, right_edges, right_gradients, search_radius):
    #Pad the edge images:
    padded_left_edges = np.pad(left_edges, (search_radius,search_radius), 'constant', constant_values=(0,0))
    padded_right_edges = np.pad(right_edges, (search_radius,search_radius), 'constant', constant_values=(0,0))

    points = np.zeros_like(left_edges)
    values = np.zeros_like(left_edges, dtype=np.float)

    #Iterate through the image:
    for x in xrange(search_radius, len(padded_left_edges)-search_radius):
        for y in xrange(search_radius, len(padded_left_edges[0])-search_radius):

            #Check if there is an edge:
            if padded_left_edges[x,y]:
                best_similarity = -1
                #Iterate through the window to see if there are other edges:
                for i in xrange(-search_radius, search_radius):
                    for j in xrange(-search_radius, search_radius):
                        if padded_right_edges[x+i,y+j]:
                            g1 = left_gradients[x-search_radius,y-search_radius]
                            g2 = right_gradients[x-search_radius+i,y-search_radius+j]

                            #Calculate cosine distance:
                            n1 = np.linalg.norm(g1)
                            n2 = np.linalg.norm(g2)

                            #We dont like zero gradients:
                            if n1 != 0 and n2 != 0:
                                #Compare it:
                                similarity = np.dot(g1, g2)/n1/n2

                                if similarity > best_similarity:
                                    points[x-search_radius,y-search_radius] = 1
                                    values[x-search_radius,y-search_radius] = math.sqrt(i**2 + j**2)
                                    best_similarity = similarity


    return points, values

'''
Stereoanalysis:
'''

#Define a stereoanalysis function using the submodules:
def analyse(left_image, right_image, pyramid_levels=4, search_radius = 10, maxitt=100, l=0.01):
    #Calculate pyramids:
    left_pyramid = Pyramids.down_pyramid(left_image, levels=pyramid_levels)
    right_pyramid = Pyramids.down_pyramid(right_image, levels=pyramid_levels)

    #Define some arrays to hold edges and gradients:
    left_edges = [None]*pyramid_levels
    left_gradients = [None]*pyramid_levels
    right_edges = [None]*pyramid_levels
    right_gradients = [None]*pyramid_levels

    #Do the calculation:
    for i in xrange(pyramid_levels):
        left_edges[i] = edge_map(left_pyramid[i])
        right_edges[i] = edge_map(right_pyramid[i])
        left_gradients[i] = gradient_map(left_pyramid[i])
        right_gradients[i] = gradient_map(left_pyramid[i])

    result_matrices = [None]*pyramid_levels

    #TODO: What are we supposed to do with the first layer?
    #Set the first prior:
    result_matrices[-1] = np.zeros_like(left_edges[-1], dtype=np.float)

    #Run through the layers, interpolating from the previous:
    for i in reversed(xrange(pyramid_levels)):
        (points, values) = match(left_edges[i], left_gradients[i], right_edges[i], right_gradients[i], search_radius)
        result_matrices[i] = Helper.interp(result_matrices[i], points, values, maxitt, l)

        if i > 0:
            result_matrices[i-1] = np.multiply(Pyramids.upsample(result_matrices[i],desired_corrected_size=left_edges[i-1].shape),2)

    #Return the interpolation at the top level:
    return result_matrices[0]

def show_disparity_map(disparity_map):
    #Scale to zero:
    image = np.subtract(disparity_map, np.min(disparity_map))

    if np.max(image) != 0:
        #Normalize and invert:
        image = np.multiply(image, -255.0/float(np.max(image)))
        image = np.add(image, 255)


    #Show the stuff:
    viewer = ImageViewer(image.astype(np.uint8))
    viewer.show()

'''
Testing:
'''

def build_match_dic(image1, image2, matching_algorithm):
    d = {}

    p,v = matching_algorithm(image1, image2)

    print image1.shape
    print image2.shape
    print p.shape
    print v.shape

    print "Converting to dictionary..."
    for x in xrange(p.shape[0]):
        for y in xrange(p.shape[1]):
            if p[x,y]:
                d[(x,y)] = v[x,y]

    return d


#Define a method to produce an evaluation of the matching strategy:
def show_matching(img, img2, matching):
    print "matching..."
    ip_match = build_match_dic(img, img2, matching)

    print "Constructing intermediate image..."
    padding = 5 #padding around the edges

    bar = np.ndarray((img.shape[0], 5))
    bar.fill(1.0)
    viewer = ImageViewer(img)
    viewer.show()
    img3 = np.column_stack((img, bar, img2))
    viewer = ImageViewer(img3)
    viewer.show()

    img3 = img_as_ubyte(img3)

    viewer = ImageViewer(img3)
    viewer.show()

    img3 = np.pad(img3, pad_width=padding, mode='constant', constant_values=(0))


    viewer = ImageViewer(img3)
    viewer.show()
    print "Drawing lines..."

    colimg = color.gray2rgb(img3)
    for k,v in random.sample(ip_match.items(), int(float(len(ip_match.keys()))*0.005)):
        #Choose a random colour:
        col = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]

        #Calculate coordinates after padding:
        x1 = k[0]+padding
        y1 = k[1]+padding
        x2 = v[0]+padding
        y2 = v[1] + img.shape[1]+bar.shape[1]+padding

        #Draw the points in both images:
        rr, cc = circle_perimeter(x1, y1, 3)
        colimg[rr, cc] = col
        rr, cc = circle_perimeter(x2, y2, 3)
        colimg[rr, cc] = col

        #Draw a line between the points:
        rr, cc = line(x1,y1,x2,y2)
        colimg[rr, cc] = col

    #Show the result:
    viewer = ImageViewer(colimg)
    viewer.show()


if __name__ == "__main__":
    limg = imread('venusL.png')
    limg2 = color.rgb2gray(limg)
    rimg = imread('venusR.png')
    rimg2 = color.rgb2gray(rimg)

    print "done loading"

    show_matching(limg2,rimg2,gradient_match_wrapper)

    #img3 = analyse(limg2, rimg2, pyramid_levels=4,maxitt=100)

    #show_disparity_map(img3)
