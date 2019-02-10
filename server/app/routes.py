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
_SEARCH_FILE_ = 100000
_RESULTS_ = 25
_EMBEDDINGS_ = None
_LAST_FILE_ = ""

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global
data = []
embed = []
embed_init = []
annoyObj = None

def generateLuckyData():
    global embed

    print("Getting magic pill")
    _data = data

    print("Chaos theory is now in motion...")
    _data = skShuffle(_data)

    print("Restricting lucky factor...")
    _data = _data.iloc[:10000]

    print("Creating matrix...")
    embed = _data.iloc[:, 3:515].as_matrix()

    buildAnnoy('lucky', 10)

    print("I'm feeling lucky is ready!")

def InitDB():
    global data, embed
    filename = '/ib/junk/junk/shany_ds/shany_proj/server/inference.csv'

    print("Loading db into memory...")
    data = pd.read_csv(filename, delimiter=',')

    print("Creating matrix...")
    embed = data.iloc[:, 3:515].as_matrix()

    print('Init Data loaded successfully.')

    loadData(_SEARCH_FILE_)
    loadAnnoy('inference')

def loadData(size = -1):
    global data, embed

    print("Switching data size..")

    if size != -1:
        _data = data.iloc[:size]
    else:
        _data = data

    print("Creating matrix...")
    embed = _data.iloc[:, 3:515].as_matrix()

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

    if _ALGO_ == 'annoy_euclidean':
        annoyObj = AnnoyIndex(embed.shape[1], metric='euclidean')
    elif _ALGO_ == 'annoy_manhattan':
        annoyObj = AnnoyIndex(embed.shape[1], metric='manhattan')
    elif _ALGO_ == 'annoy_hamming':
        annoyObj = AnnoyIndex(embed.shape[1], metric='hamming')
    elif _ALGO_ == 'annoy_dot':
        annoyObj = AnnoyIndex(embed.shape[1], metric='dot')
    else:
        annoyObj = AnnoyIndex(embed.shape[1])

    print("Loading annoy index...")
    annoyObj.load(filename)

    print('Annoy loaded successfully.')


def inferAll(embeddings=None):
    global data, embed, annoyObj

    if embeddings is None:
        print("Processing uploaded image...")
        res = ml.inferImage('')

        print("Embeddings received.")
        original_embed = res[0][2]
    else:
        original_embed = embeddings

    res = []

    if _ALGO_ == 'annoy' or _ALGO_ == 'annoy_euclidean' or _ALGO_ == 'annoy_manhattan' or _ALGO_ == 'annoy_hamming' or _ALGO_ == 'annoy_dot':
        print("Searching using KNN Annoy algorithm...")

        u_res = annoyObj.get_nns_by_vector(original_embed, _RESULTS_, -1, True)
        for i,d in zip(u_res[0], u_res[1]):
            res.append([float(d), int(data.iloc[i, 516]), data.iloc[i,0], int(data.iloc[i,515]) ])

    if _ALGO_ == 'cosine':
        print("Searching using Cosine Similarities...")

        similarities = cosine_similarity([original_embed], embed)[0]
        res = list(zip(similarities, data.iloc[:, 516], data.iloc[:,0], data.iloc[:,515]))
        res = sorted(res, reverse=True,key=lambda x: x[0])[:_RESULTS_]

    return [original_embed, res]

InitDB()

print('ShanyNet Loaded successfully.')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global _ALGO_, _SEARCH_FILE_, _SEARCH_, _RESULTS_, _EMBEDDINGS_, _LAST_FILE_

    result = {
        'data': [],
        'algorithm': _ALGO_,
        'search': _SEARCH_,
        'results': _RESULTS_,
        'msg': "done"
    }

    if request.method == 'POST':
        file = request.files['file']

        if _LAST_FILE_ != file.filename:
            _EMBEDDINGS_ = None
            _LAST_FILE_ = file.filename

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
                    _SEARCH_FILE_ = -1
                    loadData(_SEARCH_FILE_)
                    loadAnnoy('inference')
                elif _SEARCH_ == 'slow':
                    _SEARCH_FILE_ = 500000
                    loadData(_SEARCH_FILE_)
                    loadAnnoy('500k')
                elif _SEARCH_ == 'normal':
                    _SEARCH_FILE_ = 250000
                    loadData(_SEARCH_FILE_)
                    loadAnnoy('250k')
                elif _SEARCH_ == 'fast':
                    _SEARCH_FILE_ = 100000
                    loadData(_SEARCH_FILE_)
                    loadAnnoy('100k')
                elif _SEARCH_ == 'loose':
                    _SEARCH_FILE_ = 10000
                    loadData(_SEARCH_FILE_)
                    loadAnnoy('10k')
                elif _SEARCH_ == 'lucky':
                    _SEARCH_FILE_ = 'lucky'
                    generateLuckyData()
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
            res = inferAll(_EMBEDDINGS_)
            result['data'] = res[1]
            _EMBEDDINGS_ = res[0]

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
