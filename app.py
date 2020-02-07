
from fastai import *
from fastai.vision import *
import random
# from flask import Flask, redirect, url_for, request, render_template, send_file, jsonify
from gevent.pywsgi import WSGIServer
import cv2
from werkzeug.utils import secure_filename
from jinja2 import Environment, FileSystemLoader
import torch
import matplotlib.pyplot as plt

class CustomImageItemList(ImageList):
    def custom_label(self,df, **kwargs)->'LabelList':
        """Custom Labels from path"""
        file_names=np.vectorize(lambda files: str(files).split('/')[-1][:-4])
        get_labels=lambda x: df.loc[x,'lesion']
        #self.items is an np array of PosixPath objects with each image path
        labels= get_labels(file_names(self.items))
        y = CategoryList(items=labels)
        res = self._label_list(x=self,y=y)
        return res
path = Path('models')
export_file_name = 'export.pkl'

path = Path(__file__).parent

env = Environment(loader=FileSystemLoader(['./templates']))
learn = load_learner(path, export_file_name)

from sanic import Sanic, response
app = Sanic(__name__)

app.static('/static', './static')

# Any results you write to the current directory are saved as output.
@app.route('/', methods=['GET'])
def index(request):
    data = {'name': 'name'}
    template = env.get_template('index.html')
    html_content = template.render(name=data["name"])
    # Main page
    return response.html(html_content)

# Define a flask app
@app.route('/predict', methods=['GET', 'POST'])
def predict(request):
    f = request.files.get('file')
    
    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.name))
    write = open(file_path, 'wb')
    write.write(f.body)
    img = open_image(file_path)
    pred_class,pred_idx,outputs = learn.predict(img)
   
    file_name=secure_filename(f.name).split('.')[0]
    im = cv2.imread(file_path)
    plt.imshow(im)

    return response.json({
        'file_name': file_name,

        'status': str(pred_class)
    })

@app.route('/get_image')
def get_image(request):
    path = request.args.get('p')
    _, ext = os.path.splitext(path)
    exists = os.path.isfile(path)
    if exists:
        return response.file(path, mime_type='image/' + ext[1:])
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, access_log=False, workers=1)