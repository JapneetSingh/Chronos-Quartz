import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


"""
Input:
Output:
"""





####################Image Load,Display,Save#########################
def destroy():
    '''
    Destroy(close) all open image windows
    '''
    cv2.destroyAllWindows()

def display(image,title = 'Image', display_time = 2500):
    """
    Input:Image , title to be displayed in the window and argument for waitKey
    Output: None
    """

    cv2.imshow(title,image)
    cv2.waitKey(display_time)





####################Image preprocessing oprations#####################

def resize(image, hnew = 300,  interpolation = cv2.INTER_AREA,fixed_shape= True):

    '''
    Input:  Image, The new height you wish to set, interpolation technique, fixed shape boolean to represent if you want
    Output: Resized image
    '''

    #Compute new width . Aspect ratio must be unchanged. This is because we can ensure that the images are same size after masking.
    if fixed_shape:
        wnew = 200
        new_dim = (wnew,hnew)
        #print new_dim
    #Resize
        return cv2.resize(image,new_dim,interpolation = cv2.INTER_AREA)

    else :
        hold = image.shape[0]
        wold = image.shape[1]
        aspect_ratio = wold*1./hold
        wnew = int(hnew*aspect_ratio)
        new_dim = (wnew,hnew)
        return cv2.resize(image,new_dim,interpolation = cv2.INTER_AREA)

def convert_to_gray(image):
    '''

    Input: Image
    Output: Image converted to gray scale
    '''


    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


def blur(grey_image, k=3 ):
    '''

    Input: gray image
    Output: Image blurred by Gaussian method
    '''

    return cv2.GaussianBlur(grey_image,(k,k),0)

########################Edge Detection and Thresholding

def threshold(img_gray,k=7,C=4,plot = False):

    '''

    Input: gray image, k as convolution kernal(window) size,C a constant to be subtracted
    Default values for k,C were chosen based on manual tests performed on images
    Output: Image undergone adaptive threshold process
    '''
    t_image =  cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,k,C)

    if plot:
        display(t_image,"Thresholded image")
    #http://stackoverflow.com/questions/28930465/what-is-the-difference-between-/
    #flatten-and-ravel-functions-in-numpy   :Ravel is faster , memoryless
    features_threshold = t_image.ravel()
    return features_threshold


def cannyedgedetection(img_gray,t1=10,t2=180,plot = False):
    '''

    Input: Gray image, t1 is lower threshold, upper threshold
    Output: Image undergone canny edge detection
    '''
    # CANNY EDGE DETECTION

    ce_image = cv2.Canny(img_gray,t1,t2)

    if plot:
        display(ce_image,"Edged image")

    features_edge = ce_image.ravel()
    return features_edge



    #Contouring



###########################Masking#########################

def masking(image,mask_type = 'c'):

    '''

        Input: Image(can be color or gray), type of masking(c for circular and r for rectangular)
        Output: Image undergone canny edge detection

        For this project I have decided to go with cirular masks because they retain
        more information about watch while removing the background noise
        background
    '''


    #Width comes first when entering data to opencv
    cx = image.shape[1]/2
    cy = image.shape[0]/2

    center = (cx,cy)
    mask_canvas = np.zeros(image.shape[:2],dtype = 'uint8')
    white  = (255,255,255)

    #Radius should be equal to the half the image width to get the dial
    #since our watch dials are touching the image edges


    if mask_type == 'c':
        radius  = image.shape[1]/2
        cv2.circle(mask_canvas,center,radius,white,-1)
        img_masked = bitwise_operations(image,mask_canvas)
    if mask_type == 'r':
        #Since image is like a grid with top left corner as origin and width argument comes first
        #Width here is width of the entire image
       #We are going to get a rectangle which covers 2/3 of entire image size

        start_pt = (0 , cy - int(0.33*image.shape[0]))
        end_pt =  (image.shape[1],cy + int(0.33*image.shape[0]))
        cv2.rectangle(mask_canvas,start_pt,end_pt,white,-1)
        img_masked = bitwise_operations(image,mask_canvas)
    #display(img_masked,"masked")
    #Masked image can be returned if needed
    return mask_canvas #,img_masked



def bitwise_operations(image,mask):
    return cv2.bitwise_and(image,image,mask = mask)



#########################Dominant color####################


def grey_hist_feat(img_gray,mask =None):
    '''
    Input: Gray image , mask if used

    Output: features list obtain by flattening the histgram
    '''


    #img_g = convert_to_gray(image)

    img_gb = blur(img_gray,3)
    hist = cv2.calcHist([image],[0],mask,[256],[0,256])
    plt.title("Gray Histogram")
    plt.xlim([0,255])
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.show()
    #featurize
    return hist.flatten()


def col_hist_feat(image,mask = None,plot = False):
    '''
    Input: Image , mask if used, a boolean for whether to plot histogram or not

    Output: features list obtain by flattening the histgram
    '''
    # Split the image matrix to 3 seperate matrices  of B,G,R channels each now in greyscale
    bgr = cv2.split(image)
    col = ['b','g','r']
    plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")


    features = []
    for (chan, color) in zip(bgr, col):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])

        if plot:
            print "Color Histogram",hist.shape
            plt.xlim([0, 256])
            plt.plot(hist, color = color)

        #Flatten the matrix to a list
        features.extend(hist.flatten())
    if plot:
        plt.show()# We leave it at the end coz in the for loop we'll get 3 plots for b,g,r
    return  features

##########################################Preprocessing function

def preprocess(image_path):

        image = cv2.imread(image_path)

        resized_image = resize(image)

        gray_image = convert_to_gray(resized_image)


        #display(resized_image, "rz")
        #Color

        # mask_canvas = masking(resized_image , 'c')
        # display(bitwise_operations(resized_image,mask_canvas))
        # feature_color =col_hist_feat(resized_image, mask_canvas,True)
        mask_canvas = masking(resized_image , 'c')

        feature_color =col_hist_feat(resized_image,mask = mask_canvas,plot = False)
        #Edge
        feature_edge = cannyedgedetection(gray_image)
        #print (feature_edge).shape, "no of edge features"

        #Thresholding
        feature_threshold =  threshold(gray_image)
        #print (feature_threshold).shape ,"no of threshold features"

        #Combine features
        features = np.concatenate((feature_edge , feature_threshold, feature_color),0)
        #print len(features),"total features"
        return features


##############################
#Tuning Parameters
#1) Number of components in PCA for each featurizing step
#2)Threshold /Pca use gray images

##############################
##############################
if __name__ == "__main__":
    image_path = ("/Users/Iskandar/Desktop/WatchSeer/testimages/3.jpg")#Circular
    #image = cv2.imread("/Users/Iskandar/Desktop/WatchSeer/testimages/32.jpg")#Rectangular
    #print image.shape
    destroy()
    features = preprocess(image_path)