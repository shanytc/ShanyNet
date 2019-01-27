from flask import Flask
from flask_cors import CORS

UPLOAD_FOLDER = '/ib/junk/junk/shany_ds/shany_proj/dataset/inference/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__,static_folder="/ib/junk/junk/shany_ds/shany_proj/server/app/static")
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from app import routes
