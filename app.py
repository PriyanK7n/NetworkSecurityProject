import sys
import os

import certifi
ca = certifi.where() 
from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
print(mongo_db_url)


import pymongo
from NetworkSecurityProject.exception.exception import NetworkSecurityException
from NetworkSecurityProject.logging.logger import logging
from NetworkSecurityProject.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from NetworkSecurityProject.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from NetworkSecurityProject.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME


from NetworkSecurityProject.utils.main_utils.utils import load_object

from NetworkSecurityProject.utils.ml_utils.model.estimator import NetworkModel


client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)


# DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME

load_dotenv()
database = os.getenv("DATA_INGESTION_DATABASE_NAME")
collection = os.getenv("DATA_INGESTION_COLLECTION_NAME")


# Fast API Code
app = FastAPI()
origins = ["*"]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

# Jinja2Templates used to access templates folder 
from fastapi.templating import Jinja2Templates 
templates = Jinja2Templates(directory = "./templates")

# Authentication
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


# Initiation of Training Pipeline
@app.get("/train") # routes for training
async def train_route():
    try:
        train_pipeline_obj = TrainingPipeline()
        train_pipeline_obj.run_training_pipeline()
        return Response("Training is Successfull")
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)


# Initiation of Batch Prediction Pipeline (which predicts on uploaded batch of data)
@app.post("/predict") # routes for prediction
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        # Load preprocessor's transformation object and best model pickle file
        final_model_preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        # After loading pass these to NetworkModel class present in utils/ml_utils/model/estimator.py
        network_model_obj = NetworkModel(preprocessor = final_model_preprocessor, model = final_model)
        print(df.iloc[0]) #  print the first row of the DataFrame data as a pandas Series
        y_pred = network_model_obj.predict(df)
        print(y_pred)

        df['predicted_Result'] = y_pred
        print(df['predicted_Result'])
        
        df.to_csv("prediction_output/output.csv") # we're locally saving it but can also save the prediction in a database (mongo db etc)
        
        table_html = df.to_html(classes = "table table-striped")
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    app_run(app, host = "localhost", port=8000)


