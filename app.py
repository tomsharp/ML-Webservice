import pickle

from flask import Flask 
from flask_restful import Resource, Api, reqparse
from config import vect_path, model_path
from util import decompress_pickle

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
        

api.add_resource(Predictor, '/predict')

@app.route('/')
def index():
    return """
    <h1>Welcome to my document classifier!</h1>
    Submit a request to '/predict' with your hashed sentence in as the 'words' parameter.
    <br>
    For more information, please visit this project's <a href="https://github.com/tomsharp/ML-Webservice">GitHub</a>
    """

if __name__ == '__main__':
    app.run(debug=True)