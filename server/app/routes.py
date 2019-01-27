from app import app
from server import ml
from flask import request
import os
from werkzeug import secure_filename
import pandas as pd
import numpy as np
from time import time
from sklearn.metrics.pairwise import cosine_similarity
from flask import jsonify

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
NUM_OF_IMAGES_TO_SEARCH = 10

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def inferAll():
    print("Inference crunching time...")
    res = ml.inferImage('')

    print("Got embeddings!")
    original_embed = res[0][2]
    # original_embed = np.random.uniform(low=0, high=1, size=(512,))
    ### scan embeddingsa

    filename = '/ib/junk/junk/shany_ds/shany_proj/server/100k.csv'
    similatirties = []


    print("Loading embeddings into memory...")
    start = time()
    data = pd.read_csv(filename, delimiter=',')

    print("Creating matrix...")
    embed = data.iloc[:, 3:515].as_matrix()

    print("finding nearest neighbhoors...")
    similatirties = cosine_similarity([original_embed], embed)[0]

    print("Packing it up...")
    res = list(zip(similatirties, data.iloc[:, 516], data.iloc[:,0], data.iloc[:,515]))

    print("Retrieving " + str(NUM_OF_IMAGES_TO_SEARCH) + " best images...")
    res = sorted(res, reverse=True,key=lambda x: x[0])[:NUM_OF_IMAGES_TO_SEARCH]

    print("It took " + str(time() - start) + " to execute")

    return res

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    result = {
        'data': [],
        'msg': "done"
    }

    if request.method == 'POST':
        file = request.files['file']
        print(file)
        if file and allowed_file(file.filename):

            for the_file in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result['data'] = inferAll()


    #if request.method == 'GET':
    #    inferAll()
    #    return "done"


    resp = jsonify(result)
    resp.status_code = 200
    return resp


@app.route('/')
@app.route('/index')
def index():
    html = """
		<!doctype html>
		<html lang="en">
		  <head>
			<!-- Required meta tags -->
			<meta charset="utf-8">
			<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

			<!-- Bootstrap CSS -->
			<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css" integrity="sha384-GJzZqFGwb1QTTN6wy59ffF1BuGJpLSa9DkKMp0DgiMDm4iYMj70gZWKYbI706tWS" crossorigin="anonymous">

			<title>Hello, world!</title>
		  </head>
		  <body>
			<nav class="navbar navbar-light bg-light">
			  <a class="navbar-brand" href="#">
				<img src="/static/logo.png" width="30" height="30" class="d-inline-block align-top" alt="">
				ShanyNet
			  </a>
			</nav>
			<div>
				<div class="container-fluid">
				  <div class="row" style="margin-top:10px;">
				  	<div class="col-5">
					  <ul class="list-group">
						  <li class="list-group-item">
						  	<form action="/upload" method="post" enctype="multipart/form-data">
						  		<input type="file" name="file"/>
						  		<button type="submit" class="btn btn-primary">Upload</button>
						  	</form>
						  </li>
						</ul>
					</div>
					<div class="col-7">
					  right
					</div>
				  </div>
				</div>
			</div>
			<!-- Optional JavaScript -->
			<!-- jQuery first, then Popper.js, then Bootstrap JS -->
			<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
			<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.6/umd/popper.min.js" integrity="sha384-wHAiFfRlMFy6i5SRaxvfOCifBUQy1xHdJ/yoi7FRNXMRBu5WHdZYu1hA6ZOblgut" crossorigin="anonymous"></script>
			<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.2.1/js/bootstrap.min.js" integrity="sha384-B0UglyR+jN6CkvvICOB2joaf5I4l3gm9GU6Hc1og6Ls7i6U/mkkaduKaBhlAXv9k" crossorigin="anonymous"></script>
			<script src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
			<script src="https://ajax.googleapis.com/ajax/libs/angularjs/1.7.6/angular.min.js"></script>
		  </body>
		</html>
	"""

    return html
