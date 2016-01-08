import numpy as np
import cv2
import os


def resize(no_of_images, new_height, indicator=20):
	for i in xrange(no_of_images):
		
		path = "{0}/{1}.jpg".format(i,i)
		
		if os.path.exists(filename): 
			os.chdir("{0}").format(i)
			image_name ="{0}.jpg".format()
			resize(image_name,new_height,i)
		if i %	indicator == 0 :
			print "%d images resized"%i 


def resize(image_name,height_reducing_scalar,i):
	image = cv.imread(image_name)

	aspect_ratio = image.shape[0]*1./image.shape[1]
	new_width = round((aspect_ratio*height),0)

	cv2.imsave("{0}r.jpg").format(i)


if ___name__ = "__main__":
	