import pickle

from flask import Flask 
from flask_restful import Resource, Api, reqparse
import bz2
import _pickle as cPickle
from config import vect_path, model_path

def decompress_pickle(path):
    data = bz2.BZ2File(path, 'rb')
    data = cPickle.load(data)
    return data

app = Flask(__name__)
api = Api(app)

vectorizer = decompress_pickle(vect_path)
model = decompress_pickle(model_path)

parser = reqparse.RequestParser()
parser.add_argument('words')

class Predictor(Resource):
    def get(self):

        args = parser.parse_args()
        words = args['words']

        # prediction
        words_vect = vectorizer.transform([words])
        prediction = model.predict(words_vect)[0]
        
        # confidence
        probs = list(model.predict_proba(words_vect)[0])
        c_index = probs.index(max(probs))
        confidence = probs[c_index]

        return {'prediction': prediction, 'confidence':confidence}
        

api.add_resource(Predictor, '/')

if __name__ == '__main__':
    app.run(debug=False, port=5000)