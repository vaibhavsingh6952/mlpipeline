import pandas as pd
from sklearn.metrics import accuracy_score
import os
import pickle
import yaml
import mlflow
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/vaibhavsingh6952/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "vaibhavsingh6952"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "2a0f8e0c81415f5282724876f41c279be46a1fa4"

## load parameters from yaml file
params = yaml.safe_load(open("params.yaml"))['train']

def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/vaibhavsingh6952/machinelearningpipeline.mlflow")

    ## load the model
    model = pickle.load(open(model_path,'rb'))

    ## prediction, accuracy and logging them
    y_pred = model.predict(X)
    accuracy = accuracy_score(y,y_pred)
    mlflow.log_metric('accuracy',accuracy)
    print(f"Model accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate(params['data'],params['model'])


