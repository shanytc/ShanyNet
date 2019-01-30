from app import app
from server import ml
from flask import request, send_file
import os
from werkzeug import secure_filename
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import jsonify
from annoy import AnnoyIndex
from sklearn.utils import shuffle as skShuffle

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
_ALGO_ = 'annoy'
_SEARCH_ = 'fast'
_SEARCH_FILE_ = '100k'
_RESULTS_ = 25

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global
data = []
embed = []
start = 0
annoyObj = None

def generateLuckyData(file):
    global embed
    filename = '/ib/junk/junk/shany_ds/shany_proj/server/'+file+'.csv'

    print("Loading "+file+".csv into memory for luckiness...")
    _data = pd.read_csv(filename, delimiter=',')

    print("Chaos theory is now in motion...")
    _data = skShuffle(_data)

    print("Restricting lucky factor...")
    _data = _data.iloc[:10000]

    print("Creating matrix...")
    embed = _data.iloc[:, 3:515].as_matrix()

    buildAnnoy('lucky', 10)

    print("I'm feeling lucky is ready!")


def loadData(file):
    global data, embed, start
    filename = '/ib/junk/junk/shany_ds/shany_proj/server/'+file+'.csv'

    print("Loading "+file+".csv into memory...")
    data = pd.read_csv(filename, delimiter=',')

    print("Creating matrix...")
    embed = data.iloc[:, 3:515].as_matrix()

    print('Data loaded successfully.')

def buildAnnoy(file, treeSize=100):
    filename = '/ib/junk/junk/shany_ds/shany_proj/server/'+file+'.ann'

    print("Creating annoy index...")
    t = AnnoyIndex(embed.shape[1])

    i = 0
    print("Adding elements to AnnoyIndex...")
    for element in embed:
        t.add_item(i,element)
        i+=1

    print("Creating annoy tree...")
    t.build(treeSize)

    print("Saving annoy..")
    t.save(filename)

    print('Annoy indexing complete.')

def loadAnnoy(file):
    global embed, annoyObj

    filename = '/ib/junk/junk/shany_ds/shany_proj/server/'+file+'.ann'

    # use annoy
    print("Creating annoy index...")
    annoyObj = AnnoyIndex(embed.shape[1])

    print("Loading annoy index...")
    annoyObj.load(filename)

    print('Annoy loaded successfully.')

def inferAll():
    global data, embed, annoyObj

    print("Processing uploaded image...")
    res = ml.inferImage('')

    print("Embeddings received.")
    original_embed = res[0][2]

    res = []

    if _ALGO_ == 'annoy':
        print("Searching using KNN Annoy algorithm...")

        u_res = annoyObj.get_nns_by_vector(original_embed, _RESULTS_, -1, True)
        for i,d in zip(u_res[0], u_res[1]):
            res.append([float(d), int(data.iloc[i, 516]), data.iloc[i,0], int(data.iloc[i,515]) ])

    if _ALGO_ == 'cosine':
        print("Searching using Cosine Similarities...")

        similarities = cosine_similarity([original_embed], embed)[0]
        res = list(zip(similarities, data.iloc[:, 516], data.iloc[:,0], data.iloc[:,515]))
        res = sorted(res, reverse=True,key=lambda x: x[0])[:_RESULTS_]

    return res

loadData(_SEARCH_FILE_)
loadAnnoy(_SEARCH_FILE_)

print('ShanyNet Loaded successfully.')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global _ALGO_, _SEARCH_FILE_, _SEARCH_, _RESULTS_

    result = {
        'data': [],
        'algorithm': _ALGO_,
        'search': _SEARCH_,
        'results': _RESULTS_,
        'msg': "done"
    }

    if request.method == 'POST':
        file = request.files['file']

        if request.form["algo"] is not None:
            algo = request.form["algo"]
            if algo != _ALGO_:
                _ALGO_ = algo
                result['algorithm'] = _ALGO_

        if request.form["results"] is not None:
            results = int(request.form["results"])
            if results != _RESULTS_:
                _RESULTS_ = int(results)
                result['results'] = _RESULTS_

        if request.form["search"] is not None:
            search = request.form["search"]

            if search != _SEARCH_:
                _SEARCH_ = search
                result['search'] = _SEARCH_

                if _SEARCH_ == 'deep':
                    _SEARCH_FILE_ = 'inference'
                    loadData(_SEARCH_FILE_)
                elif _SEARCH_ == 'slow':
                    _SEARCH_FILE_ = '500k'
                    loadData(_SEARCH_FILE_)
                elif _SEARCH_ == 'normal':
                    _SEARCH_FILE_ = '250k'
                    loadData(_SEARCH_FILE_)
                elif _SEARCH_ == 'fast':
                    _SEARCH_FILE_ = '100k'
                    loadData(_SEARCH_FILE_)
                elif _SEARCH_ == 'loose':
                    _SEARCH_FILE_ = '10k'
                    loadData(_SEARCH_FILE_)
                elif _SEARCH_ == 'lucky':
                    _SEARCH_FILE_ = 'lucky'
                    generateLuckyData('100k')

                loadAnnoy(_SEARCH_FILE_)

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
