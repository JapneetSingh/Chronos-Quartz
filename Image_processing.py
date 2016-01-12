import numpy as np
import cv2
import os
from featurize import preprocess, display, destroy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#import matplotlib.pyplot as plt


def vectorize(data_path,no_of_images,indicator = 20):
	'''
	Takes the path, total number of folders in path and creates vector matrix

	'''
	feature_matrix = []
	#Since some of the images may not be loaded(dont exist or other errors) we
	#start a dictionary to keep track of ehich ones do and which ones dont
	index_dict = {}
	#dict_counter keeps a count of number of images successfully loaded
	dict_counter = 0

	for i in xrange(no_of_images):
		image_name = data_path + "{0}/{1}.jpg".format(i,i)
		#print image_name
		if os.path.exists(image_name):

			#os.chdir("{0}").format(i)
			#image_name ="{0}.jpg".format()

			features = preprocess(image_name)
			#print features
			feature_matrix.append(features)
			index_dict[dict_counter] = i
			dict_counter += 1
		else:
			#index_dict[i] = "Load Error:No image"
			pass

		if i %	indicator == 0 :
			print "%d images finished"%i

	data_mat =  np.array(feature_matrix)
	return data_mat , index_dict

################################pCA

def pca(data,n_components = 100,plot = False):
    '''
    Since Standard scalar expects <=2 dimensions but colored images have 3 we will be
    convert them to gray if not gray already


    Input: Gray image, number of components you wish to use, plot boolean to do the scree plot

    Output: Features from image

    '''
    #Since standard scalar works with floats we change dtype here.
    #Else we get a warning

	#data = data.astype("float32")

    scale = StandardScaler()
    img_data_scaled  =  scale.fit_transform(data)
    #print ">>>>>>>>>>>before>>>>>>", img_scaled.shape

    #features
    pca_model = PCA(n_components)
    pca_data = pca_model.fit_transform(img_data_scaled)
    #print ">>>>>>>>>>>>>after>>>>>>>", pca_data.shape
    if plot:
        # PLot to see how much variance does each principal component explain
        scree_plot(pca_model)
        plt.show()

    variance_array = pca_model.explained_variance_ratio_
    return pca_data, variance_array


############################# PCA helper
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



if __name__ == "__main__":
	data_path = 'Data/'
	data, index = vectorize(data_path,5516,20)
	data_new = data[:4000,:]
	data_pca , var = pca(data_new,800)
