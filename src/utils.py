import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path) as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        logging.info("An error has occured while saving object")
        raise CustomException(e,sys)

def evaluate_model(y_test,y_pred):
    try:
        mae = mean_absolute_error(y_test,y_pred)
        mse = mean_squared_error(y_test,y_pred)

        r2 = r2_score(y_test,y_pred)


        return mae,mse,r2
    except Exception as e:
        logging.info("An error as occured while evaluating the model")
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path) as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("An error as occured while loading the pickle object")