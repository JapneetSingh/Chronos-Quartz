import matplotlib
#matplotlib.use('Agg') # must be uncommented for use on aws
import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import cPickle as pickle


####################Image Load,Display,Save#########################
def destroy():
    '''
    Destroy(close) all open image windows from display
    Input : None
    Output: None
    '''
    cv2.destroyAllWindows()


def display(image, title='Image', display_time=2500):
    """
    Display images using opencv module
    Input:Image , title to be displayed in the window and argument for waitKey
    Output: None
    """

    cv2.imshow(title, image)
    cv2.waitKey(display_time)


def pickle_this(object, filename):
    """
    Takes in the filename(location can be included) and stores the object there
    Input: Object to be pickled, location for saving it with name of file that has a pickle extension
    Output: None
    """

    with open(filename, "wb") as fid:
        pickle.dump(object, fid)

####################Image preprocessing oprations#####################


def resize(image, hnew=300, interpolation=cv2.INTER_AREA, fixed_shape=True):
    '''
    Resizes the image
    Input:  Image, The new height you wish to set, interpolation technique, fixed shape boolean to represent if you want
    Output: Resized image
    '''

    # Compute new width . Aspect ratio must be unchanged. This is because we
    # can ensure that the images are same size after masking.
    if fixed_shape:
        wnew = 200
        new_dim = (wnew, hnew)
        # print new_dim
        # Resize
        return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

    else:
        hold = image.shape[0]
        wold = image.shape[1]
        aspect_ratio = wold * 1. / hold
        wnew = int(hnew * aspect_ratio)
        new_dim = (wnew, hnew)
        return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)


def convert_to_gray(image):
    '''
    Converts image to grayscale
    Input: Image
    Output: Image converted to gray scale
    '''

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#This function was not used
def convert_to_hsv(image):
    '''
    Converts image to hsv.
    Input: Image
    Output: Image converted to hsv
    '''

    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def blur(grey_image, k=3):
    '''
    Blur the image
    Input: gray image , kernal size to perform the blur(smaller the size lesser the blur)
    Output: Image blurred by Gaussian method
    '''

    return cv2.GaussianBlur(grey_image, (k, k), 0)

################## Edge Detection and Thresholding ####################

def threshold(img_gray, k=7, C=4, plot=False):
    '''

    Input: gray image, k as convolution kernal(window) size,C a constant to be subtracted
    Default values for k,C were chosen based on manual tests performed on images
    Output: Image matrix undergone adaptive threshold process
    '''
    t_image = cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        k,
        C)

    if plot:
        display(t_image, "Thresholded image")
    # http://stackoverflow.com/questions/28930465/what-is-the-difference-between-/
    # flatten-and-ravel-functions-in-numpy   :Ravel is faster , memoryless
    features_threshold = t_image.ravel()
    return features_threshold


def cannyedgedetection(img_gray, t1=10, t2=180, plot=False):
    '''

    Input: Gray image, t1 is lower threshold, upper threshold
    Output: Image undergone canny edge detection
    '''
    # CANNY EDGE DETECTION

    ce_image = cv2.Canny(img_gray, t1, t2)

    if plot:
        display(ce_image, "Edged image")

    features_edge = ce_image.ravel()
    return features_edge

###########################Masking#########################


def masking(image, mask_type='c'):
    '''

    Used to generate a mask for images used in color extraction.
    The circular mask covers the dial and
    the rectangluar covers the band color with a part of dial
    Input: Image(can be color or gray), type of masking(c for circular and r for rectangular)
    Output: Image undergone canny edge detection
    '''

    # Width comes first when entering data to opencv
    cx = image.shape[1] / 2
    cy = image.shape[0] / 2

    center = (cx, cy)
    mask_canvas = np.zeros(image.shape[:2], dtype='uint8')
    # if hsv white  = (0,0,1)

    white = (255, 255, 255)
    # Radius should be equal to the half the image width to get the dial
    # since our watch dials are touching the image edges

    # used for dial color features
    if mask_type == 'c':
        radius = image.shape[1] / 2
        cv2.circle(mask_canvas, center, radius, white, -1)
        img_masked = bitwise_operations(image, mask_canvas)

    # used for Band color features
    if mask_type == 'r':
        # Since image is like a grid with top left corner as origin and width argument comes first
        # Width here is width of the entire image
        # We are going to get a rectangle which the length of the watch which
        # covers the band and samll portion of dial to get dominant colors

        start_pt = (cx - (image.shape[1] / 8), 0)
        end_pt = (cx + (image.shape[1] / 8), int(image.shape[0]))
        cv2.rectangle(mask_canvas, start_pt, end_pt, white, -1)
        img_masked = bitwise_operations(image, mask_canvas)
    # display(img_masked,"masked")
    # Masked image can be returned if needed
    return mask_canvas  # ,img_masked


def bitwise_operations(image, mask):
    """
    Perform a bitwise and operation to apply the mask to the image
    Input: Image,mask
    Output:Masked image
    """

    return cv2.bitwise_and(image, image, mask=mask)

#########################Dominant color####################

#This function was not used
def grey_hist_feat(img_gray, mask=None):
    '''
    Returns grey scale based features for gray scaled image
    Input: Gray image , mask if used

    Output: features list obtain by flattening the histgram
    '''
    #img_g = convert_to_gray(image)
    img_gb = blur(img_gray, 3)
    hist = cv2.calcHist([image], [0], mask, [256], [0, 256])

    # Used to give a different color to the labels/title of graphs
    font = {'family': 'serif',
            'color': 'white',
            'weight': 'normal',
            'size': 16,
            }

    plt.title("Gray Histogram", fontdict=font)
    plt.xlim([0, 255])
    plt.xlabel("Bins", fontdict=font)
    plt.ylabel("# of Pixels", fontdict=font)
    plt.plot(hist)
    plt.show()
    # featurize
    return hist.flatten()


def col_hist_feat(image, mask=None, plot=False):
    '''
    Returns RGB based features for color images
    Input: Image , mask if used, a boolean for whether to plot histogram or not

    Output: features list obtain by flattening the histgram
    '''
    # Split the image matrix to 3 seperate matrices  of B,G,R channels each
    # now in greyscale
    bgr = cv2.split(image)
    col = ['b', 'g', 'r']

    # Preparing to plot if needed
    # Used to give a different color to the labels/title of graphs
    font = {'family': 'serif', 'color': 'Red', 'weight': 'normal', 'size': 16}

    plt.figure()
    plt.title("Flattened Color Histogram", fontdict=font)
    plt.xlabel("Bins", fontdict=font)
    plt.ylabel("# of Pixels", fontdict=font)

    features = []
    for (chan, color) in zip(bgr, col):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])

        if plot:
            print "Color Histogram", hist.shape
            plt.xlim([0, 256])
            plt.plot(hist, color=color)

        # Flatten the matrix to a list
        features.extend(hist.flatten())
    if plot:
        plt.show()  # We leave it at the end coz in the for loop we'll get 3 plots for b,g,r
    return features

# Preprocessing function


def preprocess(image_path):
    """

    Takes in a single image, resizes it and featurizes it
    using various techniques(edge detection, thresholding and histograms)
    Input: An image
    Output: Returns a matric constituting derived from the image
    """

    # Read in the image and resize it
    image = cv2.imread(image_path)
    resized_image = resize(image)

    # Get hsv features for image. Continuing with RGB for now
    #resized_image = convert_to_hsv(resized_image)

    # Get the color features for the dial
    mask_canvas_dial = masking(resized_image, 'c')
    feature_color_dial = col_hist_feat(
        resized_image, mask=mask_canvas_dial, plot=False)

    # Get the color features for the bands and part of dial(across the length
    # of the watch)
    mask_canvas_band = masking(resized_image, 'r')
    feature_color_band = col_hist_feat(
        resized_image, mask=mask_canvas_band, plot=False)

    # Combine the two color features to make one
    feature_color = np.concatenate((feature_color_band, feature_color_dial), 0)

    #Edge and thresholding
    gray_image = convert_to_gray(resized_image)
    # Edge
    feature_edge = cannyedgedetection(gray_image)
    #print (feature_edge).shape, "no of edge features"

    # Thresholding
    feature_threshold = threshold(gray_image)
    #print (feature_threshold).shape ,"no of threshold features"

    features = np.concatenate(
        (feature_threshold, feature_edge, feature_color), 0)
    return features


##############################
if __name__ == "__main__":
    image_path = (
        "/Users/Iskandar/Desktop/WatchSeer/Data/479/479.jpg")
    destroy()
    features = preprocess(image_path)
