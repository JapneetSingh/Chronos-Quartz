import numpy as np
from sklearn.neighbors import NearestNeighbors
from Image_processing import vectorize, pca
from pymongo import MongoClient


#Index Dictionary maintains the mapping of images in the hardisk and their number in the data matrix
# The numbers ideally should be same but since some of the images cant be loaded there ends up being a mismatch
#between the two. Once we have the predicted indices based on model we can use them to get the real indices on hard disk
#which is the referene number also used in Mongodb
def get_results(index_dict,prediction_indices,test):
    '''
    Input: Index dictionary , Prediction indices recommended
    Output: list of links to recommended products
    '''
    client = MongoClient()
    db = client.images  #db
    source_data = db.image_data #image

    x = source_data.find({'img_no' : test})
    x.rewind()


    results = []
    for ind in prediction_indices:
        img_number = index_dict[ind]
        data =source_data.find({'img_no':img_number})
        data.rewind()
        results.append(str(data[0]['prod_url']))
    return results


# User image will become a link in the next iteration. Number for now

def model(user_image, var_explained = False , data_paths = 'Data/'):

    '''
    #perform pca for the data
    '''

    # we get data from reading the images
    data,index_dict = vectorize(data_paths,5516,25)

    #Keeping only 4000 becasue PCA is breaking for bigger numbers
    data_new = data[:4000,:]

    '''
    #perform pca for the data
    '''
    data_pca,variance_per_component = pca(data_new,800)

    if var_explained:
        print "Cumulative sum of Variance per component is as follows:" , variance_per_component.cumsum()


    model =  NearestNeighbors(n_neighbors = 10)

    model.fit(data_pca)
    test_data = data_pca[user_image,:]
    if not test_data.shape[1]:
        test_data.reshape(1,-1)
    #Predict K nearest neighbors for the given vector
    distances,predicted_indices = model.kneighbors(data_pca[user_image,:])

    recommended_watches = get_results(index_dict,predicted_indices,user_image)

    print recommended_watches

if __name__ == "__main__":


    model(3976,True)
