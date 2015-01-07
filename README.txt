README

	This program contains an implementation of a feature-based stereovision algorithm relying on a pixel-by-pixel
	gradient orientation matching strategy. It is written using the Anaconda distribution of Python 2.7, and 
	utilizes the scikit-image for displaying images, computing gradients, and extracting edges. The program also
	relies heavily on numpy for algebraic calculations.
	
INSTRUCTIONS	

	The program should be accessed through the main file "StereoAnalyser.py", either from the command line with "python
	StereoAnalyser.py", or imported as a module.
	
	If run in the command line, a small text interface is provided. Here, the user is allowed to select a left and right
	image, from which a disparity map will be computed.	If imported as a module, the functionality should be accessed 
	through the function "analyse".
	
	The library file Pyramids.py contains various functions related to upsampling and downsampling with a Gaussian filter.
	Furthermore, a translation of the Thin-Plate Spline interpolation algorithm written by Søren Olsen can be found in 
	Helper.py.

CONTACT
	
	If there is any problem, please contact me (Michael Schlichtkrull) through my student mail qwt774@alumni.ku.dk.
