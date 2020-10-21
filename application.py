import pickle

from flask import Flask 
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

vect_path = 'modeling/vectorizer.pkl'
model_path = 'modeling/model.pkl'
with open(vect_path, 'rb') as f:
    vectorizer = pickle.load(f)
with open(model_path, 'rb') as f:
    model = pickle.load(f)

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