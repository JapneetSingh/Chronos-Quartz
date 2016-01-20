from collections import Counter
import os
import sys
from flask import Flask, request, render_template
from Query_Image import query_image_pipeline, unpickle
from Model import read_mongodump


app = Flask(__name__)


# Get the image metadat to be used to print results
mongo_data = read_mongodump("images.json")


# Uploading Image based models

pca_model = unpickle("pickled_models/Image/2Pca_Image_model.pkl")
scale_model = unpickle("pickled_models/Image/2SS_model.pkl")
knn_model = unpickle("pickled_models/Image/2knn_model.pkl")
image_index_dict = unpickle("pickled_models/Image/2Image_model_Index_dict.pkl")



# Home page of the web app
@app.route('/')
def submission_page():
	"""
    Home page used to collect information
    Input: None
    Output: html code

    """

    return '''
        <!DOCTYPE html>
		<html>
		<head><p align = 'center'><font  size = '6' color = #FF539E> CHRONOS&QUARTZ </font></p><BR><br><br>
		</head>

		<body link = 'red' vlink="#FF4500"  style="background-color:#1E3C6A;" background = "http://goo.gl/hEh3Xd">
		<form action = "/watch_recommendations"	method='POST'>


		<p ><font color = "white" size = 2>

		Please not that for successful results the picture of the watch should be taken in a way similar to the picture shown in the
		<a href = http://goo.gl/Ei1Ux8 color = "Grey">link</a>
		<br>
		<br>

		Url: <br>
		<input type="text" name="url"value="http://ecx.images-amazon.com/images/I/91LfiWuBpKL._UY879_.jpg"><br><br>
		<input type="submit" value="Submit">
		</font>
		</form>
		</body>
		</html>
        '''


# Results page of the app
@app.route('/watch_recommendations', methods=['POST', 'GET'])
def watch_recos():
    """
    Extracts the url from home page and displays the recommendations
    Input: None
    Output: html code

    """


    image_path = str(request.form['url'])
    results = query_image_pipeline(
        image_path,
        mongo_data,
        scale_model,
        pca_model,
        knn_model,
        image_index_dict)
    links = []
    for r in results:
        links.append(r['prod_url'])

    #text = str(request.form['desc'])

    return render_template("results.html", page_links=links)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
