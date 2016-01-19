import numpy as np
import os
from urllib import urlretrieve
import cPickle as pickle
from featurize import preprocess,pickle_this


def unpickle(filename):
	"""
	Input: Takes in a pkl file
	Output: Returns the pickled object
	"""


	with open(filename, 'rb') as fid:
		model_object = pickle.load(fid)
	return model_object


def get_results(index_dict,prediction_indices,mongo_data):
    '''
    Used the mappingin index dictioanry to get the correct image numbers. It then uses the Mongodb dump to
    get recommendations's amazon url and image
    Input: Index dictionary , Prediction indices recommended, and mongo dump as dictionary
    Output: list of dictionaries containing image and prod links to recommended products
    '''
    results = []
    #Prediction indices is an array of array with just one value inside
    for ind in prediction_indices[0]:
        #Get the original image number from index dictionary
        img_number = index_dict[ind]
        #Now that we have the original image number we proceed with getting the desired metadata for them
        #mongo_data id a defaultdict object -- a dictionary of dictionaries
        # We store every single output as dictionary
        recommend = {}
        recommend['prod_url'] = mongo_data[img_number]["prod_url"]
        recommend['reco_image_url'] = mongo_data[img_number]["img_url"]
        recommend['pred_index'] = ind
        recommend['original_img_no'] = img_number
        results.append(recommend)

    return results

def query_image_pipeline(image_url,mongo_data, scale_model, pca_model, knn_model, image_index_dict,\
 				k = 10, filename="Query_picture.jpg"):
	"""
	Input: The url of the image provided by the user , the filename to store image as , no of predictions to be returned
	Output:

	"""
	#Save the image onto the system in the current working directory
	urlretrieve(image_url,filename)

	# get the image features using the preprocess function from
	features = preprocess(filename)
	print "Features here:??????????????", features.shape	
	#Get the PCA model and Index dictionary from pkl file which should be in the same folder as this

	scaled_data =  scale_model.transform(features)
	test_data = pca_model.transform(scaled_data)
	#Its currently arow vector. We convert it to a matrix of order 1 row*121000 columns
	test_data = test_data.reshape(1,-1)

	#Getting predictions and results

	predicted_indices = knn_model.kneighbors(test_data, n_neighbors = k ,return_distance=False)
	recommendations = get_results(image_index_dict,predicted_indices,mongo_data)

	#recommendations here is a list of dictionaries for each recommendation dictionary

	return recommendations



'''

def query_metadata_model(text,nums = {'band_width':22, 'water_resistant_depth':289, 'case_diameter':40, 'case_thickness':12},k =15 ,mongopath = "images.json" ):
	"""Input the text from """

	tfidf_model = unpickle("pickled_models/TFIDF_vectorizer.pkl")
	cat_feat = tfidf_model.transform(text)

	order_of_nums = ['band_width', 'water_resistant_depth', 'case_diameter', 'case_thickness']
	num_feat = []

	for o in order_of_nums:
		num_feat.append(nums[o])

	num_feat = np.array(num_feat)

	test_data = np.concatenate((num_feat,cat_feat),0)


	nn_index_dict = unpickle("pickled_models/Metadat_Model_Index_Dictionary.pkl")
	nn_model =  unpickle("pickled_models/Metadata_model.pkl")


	predicted_indices = neighbour_model.kneighbors(test_data, n_neighbors = k ,return_distance=False)

	#Remember predicted indice numbers dont match for the two models. Hence we compare the actual image numbers in both cases

	mongo_data = read_mongodump(mongopath)

	#List of result dictionaries
	results = get_results(nn_index_dict,predicted_indices,mongo_data)

	watch_nos = []
	for d in results:
		pass

	return images




def compare_models(image_recos, text_recos):
	"""
	Input : Predcitons form both image model and text modeling
	Output : Results in weighted order

	Idea is if some results from image model also occur in text model those are presented higher up in order.
	This is because text can ensure that user preferences are given a higher weight
	"""


	weighted_recos = []
	recos = []
	for prediction in image_recos:
		if prediction in text_recos:
			weighted_recos.append(prediction)
		else:
			recos.append(prediction)


	final_order = weighted_recos + recos
	return final_order



def watch_recommendations():
	"""
	This function is the controller and makes call to all other functions



	"""
	recom_image_dict = query_image_pipeline(image_url)

	image_recos = []
	for d in recom_image_dict:
		image_recos.append(recom_image_dict[''])...............................


	recom_text_dict = query_metadata_model(text)
	text_recos = []
	for d in recom_text_dict:
		text_recos.append(recom_text_dict['']).................................


	final_recos = compare_models(image_recos,text_recos)

	result_links = []
	for rec in final_recos:
		result_links.append(recom_image_dict[rec]['prod_url'])

	#Return top 5
	return result_links[:5]



if __name__ = "__main__":


	text = "blue dial round shaped brown strap"

'''


if __name__ == "__main__":

	image_path = 'http://ecx.images-amazon.com/images/I/91KfW521riL._UL1500_.jpg'# "http://ecx.images-amazon.com/images/I/91LfiWuBpKL._UL1500_.jpg"
	results  = query_image_pipeline(image_path)
	links = []
	for r in results:
		links.append(r['prod_url'])

	print links
