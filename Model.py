import numpy as np
from sklearn.neighbors import NearestNeighbors
from Image_processing import vectorize, pca
from featurize import pickle_this
import cPickle as pickle
import json
from collections import defaultdict


def read_mongodump(path_to_mongodump):
    """
    Takes in mongo data as a list of dictionaries.
    Returns one dictionary that has where img numbers are keys and their
    metadata dictionary is stored as values

    Input: Path to Mongodump
    Output: Returns a dictionary created from the dump
    """
    mongo_dict = []
    with open(path_to_mongodump) as f:
        for line in f:
            mongo_dict.append(json.loads(line))

    # mongo_data is now a list of dictionaries
    mongo_data = defaultdict(dict)
    for d in mongo_dict:
        img_no = d['img_no']
        mongo_data[img_no] = d

    return mongo_data


# Index Dictionary maintains the mapping of images in the hardisk and their
# number in the data matrix. The numbers ideally should be same but since some
# of the images cant be loaded there ends up being a mismatch between the two.
# Once we have the predicted indices based on model we can use them to get the
# real indices on hard disk which is the referene number also used in Mongodb
# dump called mongo_data here
def get_results(index_dict, prediction_indices, mongo_data):
    '''
    Used the mappingin index dictioanry to get the correct image numbers.
    It then uses the Mongodb dump to get recommendations's amazon url and image

    Input: Index dictionary , Prediction indices recommended, and mongo dump as dictionary
    Output: list of dictionaries containing image and prod links to recommended products

    '''
    results = []
    # Prediction indices is an array of array with just one value inside
    for ind in prediction_indices[0]:
        # Get the original image number from index dictionary
        img_number = index_dict[ind]
        # Now that we have the original image number we proceed with getting the desired metadata for them
        # mongo_data id a defaultdict object -- a dictionary of dictionaries
        # We store every single output as dictionary
        recommend = {}
        recommend['prod_url'] = mongo_data[img_number]["prod_url"]
        recommend['reco_image_url'] = mongo_data[img_number]["img_url"]
        recommend['pred_index'] = ind
        recommend['original_img_no'] = img_number
        results.append(recommend)

    return results


def model(data, test_image_vector, distance_metric="cosine"):
    '''
    Perform Modeling on data by sending the image through the pipeline
    Input: final data , index dictionary,distanc metric to be used
    Output:
    '''
    model = NearestNeighbors(
        n_neighbors=10,
        metric='cosine',
        algorithm='brute')
    model.fit(data)
    pickle_this(model, "2knn_model.pkl")
    # Predict K nearest neighbors for the given vector
    predicted_indices = model.kneighbors(
        test_image_vector, return_distance=False)
    return predicted_indices


def analysis_pipeline(test_image, mongopath="images.json", data_paths='Data/'):
    """
    Analysis pipeline is the key function.
    It get the test image vector and uses it to perform the entire process together

    """
    # we get data from reading the images:
    # vectorize(data_path,no_of_images,indicator = 20)
    print "Getting data and Index"
    data, index_dict = vectorize(data_paths, 5940, 300)

    # For now we're using an image from the dataset to test against. But test vector will be provided by user
    # Reshape is needed to ensure its a matrix and not just a row vector

    test_image_vector = data[test_image, :].reshape(1, -1)
    print "Modeling time!!"

    predicted_indices = model(data, test_image_vector)
    mongo_data = read_mongodump(mongopath)

    print "Getting results"
    recommendations = get_results(index_dict, predicted_indices, mongo_data)
    return recommendations

if __name__ == "__main__":

    test_image_number = 389
    recommendations = analysis_pipeline(test_image_number)
    print recommendations
