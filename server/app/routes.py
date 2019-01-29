from app import app
from server import ml
from flask import request, send_file
import os
from werkzeug import secure_filename
import pandas as pd
import numpy as np
from time import time
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from flask import jsonify
from annoy import AnnoyIndex

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
NUM_OF_IMAGES_TO_SEARCH = 10

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

filename = '/ib/junk/junk/shany_ds/shany_proj/server/inference.csv'
print("Loading embeddings into memory...")
start = time()
data = pd.read_csv(filename, delimiter=',')

print("Creating matrix...")
embed = data.iloc[:, 3:515].as_matrix()

# build annoy
#t = AnnoyIndex(embed.shape[1])
#i=0
#print("adding elements to annoy...")
#for element in embed:
#    t.add_item(i,element)
#    i+=1

#print("Creating annoy tree...")
#t.build(10) # 10 trees

#print("saving annoy..")
#t.save('inference.ann')

#print('Annoy indexing complete...')

# use annoy
print('Loading annoy db...')
u = AnnoyIndex(embed.shape[1])
u.load('inference.ann')

print("all done!")

def inferAll():
    print("Inference crunching time...")
    res = ml.inferImage('')

    print("Got embeddings!")
    original_embed = res[0][2]
    ### scan embedding


    u_res = u.get_nns_by_vector(original_embed, 10, -1, True)

    res = []
    for i,d in zip(u_res[0], u_res[1]):
        res.append([float(d), int(data.iloc[i, 516]), data.iloc[i,0], int(data.iloc[i,515]) ])

    return res
    res = []
    similatirties = []

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

@app.route('/image/<path>/<img>', methods=['GET'])
def image(path, img):
    filename = "/ib/junk/junk/shany_ds/shany_proj/server/app/static/" + str(path) + '/' + str(img)
    return send_file(filename, mimetype='image/jpeg')

@app.route('/')
def index():
    return "nothing to see here"
