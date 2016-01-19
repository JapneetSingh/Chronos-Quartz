from collections import Counter
import os,sys
from flask import Flask, request , render_template
from Query_Image import query_image_pipeline,unpickle
from Model import read_mongodump


app = Flask(__name__)


#Get the image metadat to be used to print results
mongo_data = read_mongodump("images.json")


#Image based models

pca_model = unpickle("pickled_models/Image/2Pca_Image_model.pkl")
scale_model = unpickle("pickled_models/Image/2SS_model.pkl")
knn_model = unpickle("pickled_models/Image/2knn_model.pkl")
image_index_dict = unpickle("pickled_models/Image/2Image_model_Index_dict.pkl")


#Metadata based models








# Form page to submit text
@app.route('/')
def submission_page():
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

		Remaining inputs are all optional and are already filled with average values<br><br>

		Description: <br>
		<input type="text" name="desc" value=""><br><br>
		Band Width(in mm): <br>
		<input type="text" name="band_wid" value="22"><br><br>

		Case Diameter(in mm): <br>
		<input type="text" name="case_dia" value="40"><br><br>


		Case Width(in mm): <br>
		<input type="text" name="case_wid" value="12"><br><br>

		Water Resistant Depth(in feet): <br>
		<input type="text" name="wat_dr" value="289"><br><br>

		<input type="submit" value="Submit">
		</font>
		</form>
		</body>
		</html>
        '''


# My word counter app
@app.route('/watch_recommendations', methods=['POST','GET'] )
def watch_recos():
    image_path = str(request.form['url'])
    results  = query_image_pipeline(image_path,mongo_data, scale_model, pca_model, knn_model, image_index_dict)
    links = []
    for r in results:
    	links.append(r['prod_url'])

    #text = str(request.form['desc'])

    return render_template("results.html", page_links = links)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
