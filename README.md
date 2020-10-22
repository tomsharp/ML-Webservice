# ML-Webservice
This repository deploys a Flask API that returns a prediction from a document classifier.
<br><br>

## Quick Start - How to use this repo
### Set up conda environment
```
$ conda create --name ml_webservice --y
$ conda activate ml_webservice
$ pip install -r requirements.txt
```

### Run the app
```
$ python app.py
```

### Test the API
To test the API and the underlying model, replace the value of the `words` parameter in the test.py file, and run the script. Alternatively, you can write your own request function via Python, cURL, etc. 
```
$ python test.py
```
<br>

## Build a new model
To build a new model, update the `models` dictionary within the *model.py* file. Additionally, you can update the `data_path` and `gs_scoring_critera` variables. You can also update the vectorizers chosen as well. Then, run the script to train and test all of the models in the `models` dictionary. The code will automatically select the best model based on your scoring criteria and save it (along with the vectorizer) in the *modeling/* folder. 
```
$ python model.py
```