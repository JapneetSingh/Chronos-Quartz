import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



####################Image Load,Display,Save#########################
def destroy():
    cv2.destroyAllWindows()

def display(image,title = 'Image', display_tim = 2500):
    cv2.imshow(title,image)
    cv2.waitKey(display_tim)





####################Image preprocessing oprations#####################

def resize(image, hnew = 300, interpolation = cv2.INTER_AREA):
    #Compute new width . Aspect ratio must be unchanged
    hold = image.shape[0]
    wold = image.shape[1]
    aspect_ratio = wold*1./hold
    wnew = int(hnew*aspect_ratio)
    new_dim = (wnew,hnew)
    #Resize
    return cv2.resize(image,new_dim,interpolation = cv2.INTER_AREA)



def convert_to_gray(image):
    '''

    Input: image
    Output: Image converted to gray scale
    '''


    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


def blur(grey_image, k=3 ):
    '''

    Input: gray image
    Output: Image blurred by Gaussian methso
    '''

    return cv2.GaussianBlur(grey_image,(k,k),0)

########################Edge Detection and Thresholding

def threshold(img_gray,k=7,C=4):

    '''

    Input: gray image, k as convolution kernal(window) size,C a constant to be subtracted
    Default values for k,C were chosen based on manual tests performed on images
    Output: Image undergone adaptive threshold process
    '''
    t_image =  cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,k,C)
    features_threshold = pca(t_image)
    return features_threshold.flatten()


def cannyedgedetection(img_gray,t1=10,t2=180):
    '''

    Input: gray image, t1 is lower threshold, upper threshold
    Output: Image undergone canny edge detection
    '''
    # CANNY EDGE DETECTION

    image_ce = cv2.Canny(img_gray,t1,t2)
    feature_edge = pca(image_ce)
    return feature_edge.flatten()



    #Contouring



###########################Masking#########################

def masking(image,mask_type = 'c'):

    '''

        Input: image(can be color or gray), type of masking(c for circular and r for rectangukar)
        Output: Image undergone canny edge detection
    '''


    #Width comes first when entering data to opencv
    cx = image.shape[1]/2
    cy = image.shape[0]/2

    center = (cx,cy)
    mask_canvas_maks = np.zeros(image.shape[:2],dtype = 'uint8')
    white  = (255,255,255)

    #Radius should be equal to the half the image width to get the dial
    #since our watch dials are touching the image edges


    if mask_type == 'c':
        radius  = image.shape[1]*2/3
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

    return img_masked,mask_canvas



def bitwise_operations(image,mask):
    return cv2.bitwise_and(image,image,mask = mask)



#########################Dominant color####################




def grey_hist_feat(img_gray,mask =None):
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
    bgr = cv2.split(image)
    col = ['b','g','r']
    plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")


    features = []
    for (chan, color) in zip(bgr, col):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        print ">>>>>>>>%s>>>>>"%color,plot
        if plot:
            print "me here",hist.shape
            plt.xlim([0, 256])
            plt.plot(hist, color = color)

        features.extend(hist)
    plt.show()
    return  features

################################pCA
def pca(img_gray,n_components = 100,plot = False):
    #Since Standard scalar expects <=2 dimensions but colored images have 3 we will be
    #convert them to gray if not gray already


    scale = StandardScaler()
    img_scaled  =  scale.fit_transform(img_gray)

    pca_model = PCA(n_components=100)
    #features
    pca_data = pca_model.fit_transform(img_scaled )
    if plot:
        scree_plot(pca_model)
        plt.show()

    variance_array = pca_model.explained_variance_ratio_
    return pca_data



############################# PCA helper
def scree_plot(pca, title=None):
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6), dpi=250)
    ax = plt.subplot(111)
    ax.bar(ind, vals, 0.35,
           color=[(0.949, 0.718, 0.004),
                  (0.898, 0.49, 0.016),
                  (0.863, 0, 0.188),
                  (0.694, 0, 0.345),
                  (0.486, 0.216, 0.541),
                  (0.204, 0.396, 0.667),
                  (0.035, 0.635, 0.459),
                  (0.486, 0.722, 0.329),
                 ])

    for i in xrange(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)

    ax.set_xticklabels(ind,
                       fontsize=12)

    ax.set_ylim(0, max(vals)+0.05)
    ax.set_xlim(0-0.45, 8+0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)

    if title is not None:
        plt.title(title, fontsize=16)
###################################
def preprocess(image):
        resized_image = resize(image)
        gray_image = convert_to_gray(resized_image)



        #get features
        features = []

        #Color
        #mask_canvas_rect = masking(resized_image , 'r')
        #mask_canvas = masking(resized_image , 'c')
        #feature_color =


        #Edge
        feature_edge = cannyedgedetection(gray_image)


        #Thresholding
        feature_threshold =  threshold(gray_image)



        return features.extend(feature_color+feature_edge+feature_threshold)





##############################
#Tuning Parameters
#1) Number of components in PCA for each featurizing step
#2)Threshold /Pca use gray images

##############################
##############################
if __name__ == "__main__":
    #image = cv2.imread("/Users/Iskandar/Desktop/Project/testimages/9.jpg")#Circular
    image = cv2.imread("/Users/Iskandar/Desktop/Project/testimages/459.jpg")#Rectangular
    circ_mask,rect_mask, new_image = preprocess(image)

    destroy()

    display(new_image,"resized_image",2500)

    display(circ_mask, "mask_circ",2500)
    display(rect_mask, "mask_rect",2500)


    features = col_hist_feat(imgc,rect_mask,True)

    grey_hist_feat(image)
    pca(image)
