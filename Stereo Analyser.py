__author__ = 'Michael'

import Pyramids
import GradientCalculator
import EdgePointDetector
import numpy as np
import Helper

'''
Matching algorithm:
'''

'''
Stereoanalysis:
'''

#Define a stereoanalysis program function using the submodules:
def analyse(left_image, right_image, matching_heuristic, pyramid_levels=4, search_radius = 2, maxitt=100, l=0.1):
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
        left_edges[i] = EdgePointDetector.detect(left_pyramid[i])
        right_edges[i] = EdgePointDetector.detect(right_pyramid[i])
        left_gradients[i] = GradientCalculator.calculate(left_pyramid[i])
        right_gradients[i] = GradientCalculator.calculate(left_pyramid[i])

    result_matrices = np.zeros_like(left_pyramid)

    #TODO: What are we supposed to do with the first layer?

    #set first prior

    #Run through the layers, interpolating from the previous:
    for i in reversed(xrange(pyramid_levels)):
        (points, values) = matching_heuristic(left_edges[i], left_gradients[i], right_edges[i], right_gradients[i], search_radius)
        result_matrices[i] = Helper.interp(result_matrices[i], points, values, maxitt, l)
        if i > 0:
            result_matrices[i-1] = Pyramids.upsample(result_matrices[i])

    #Return the interpolation at the top level:
    return result_matrices[0]

'''
Testing:
'''

if __name__ == "__main__":
    q = np.array([1,2,3,4,5])
    print q

    for i in reversed(xrange(len(q))):
        print q[i]

