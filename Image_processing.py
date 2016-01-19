import numpy as np
import cv2
import os
from featurize import preprocess, display, destroy, pickle_this
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cPickle as pickle
#import matplotlib.pyplot as plt

# pCA


def pca(
        data,
        n_components=100,
        filename="pca_image_model.pkl",
        plot=False,
        mayipickle=False):
    '''
    Since Standard scalar expects <=2 dimensions but colored images have 3 we will be
    convert them to gray if not gray already

    Input: Gray image, number of components you wish to use, plot boolean to do the scree plot

    Output: Features from image

    '''
    # Since standard scalar works with floats we change dtype here.
    # Else we get a warning

    data = data.astype("float64")

    scale = StandardScaler()
    img_data_scaled = scale.fit_transform(data)
    print ">>>>>>>>>>>before pca scaled>>>>>>", img_data_scaled
    pickle_this(scale, "SS_model.pkl")

    # features
    pca_model = PCA(n_components)
    pca_data = pca_model.fit_transform(img_data_scaled)

    if mayipickle:
        pickle_this(pca_model, filename)

    # print ">>>>>>>>>>>>>after>>>>>>>", pca_data.shape
    if plot:
        # PLot to see how much variance does each principal component explain
        scree_plot(pca_model)
        plt.show()

    variance_array = pca_model.explained_variance_ratio_
    return pca_data, variance_array


############################# PCA helper- Not used currently #############
def scree_plot(pca, title=None):
    '''
    Creates a plot of top 9 Principal Component and the variance explained by them

    Input: Pca object(model)

    Output: None

    '''

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
        ax.annotate(
            r"%s%%" %
            ((str(
                vals[i] *
                100)[
                :4])),
            (ind[i] +
             0.2,
             vals[i]),
            va="bottom",
            ha="center",
            fontsize=12)

    ax.set_xticklabels(ind,
                       fontsize=12)

    ax.set_ylim(0, max(vals) + 0.05)
    ax.set_xlim(0 - 0.45, 8 + 0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)

    if title is not None:
        plt.title(title, fontsize=16)

###########################Pipeline##################


def data_pca_pipeline(data):
    """
    Create the final dataset to be modeled. Run Pca on them and then combine them
    Input : Feature datasets of same number of rows.
    Output : Concatenated feature matrix
    """
    print "Starting PCA for edge and threshold"
    data, var_exp = pca(data, 3500, 'Pca_Image_model.pkl', mayipickle=True)
    print "Cumulative sum of Variance explained per component is as follows:", var_exp.cumsum()
    print data

    return data

######################Vectorize function##############


def vectorize(data_path, no_of_images, indicator):
    '''
    Takes the path, total number of folders(not images coz folders may be empty) in path and creates vector matrix
    Input: Path to files, Total Images, print step(how many images till notification)
    Output: Final dataset to be used in modeling after PCA ,Index dictionary
    '''

    # Since some of the images may not be loaded(dont exist or other errors) we
    # start a dictionary to keep track of which ones do and which ones dont
    index_dict = {}
    # dict_counter keeps a count of number of images successfully loaded
    dict_counter = 0

    predata = []

    for i in xrange(no_of_images):
        image_name = data_path + "{0}/{0}.jpg".format(i)

        if os.path.exists(image_name):
            # Get features
            features = preprocess(image_name)

            # Forming three matrices from all the images. these will be
            # combined later
            predata.append(features)

            index_dict[dict_counter] = i
            dict_counter += 1
        else:
            #index_dict[i] = "Load Error:No image"
            pass

        if i % indicator == 0:
            print "%d images finished" % i

    # Convert the two lists to np arrays
    data = np.array(predata)
    print data
    print">>>>>>>>>>>>>>", data.shape

    pickle_this(data, "Final_Feature_Matrix_For_PCA.pkl")
    pickle_this(index_dict, "Image_model_Index_dict.pkl")

    # Perform PCA
    data = data_pca_pipeline(data)

    return data, index_dict


###################################


if __name__ == "__main__":

    data_path = 'Data/'

    data_color, data_thed, index_dict = vectorize(data_path, 5940, 250)

    data = data_pipeline(data_color, data_thed)
