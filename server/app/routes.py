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

    resp = jsonify(result)
    resp.status_code = 200
    return resp


@app.route('/')
@app.route('/index')
def index():
    return "nothing to see here"
