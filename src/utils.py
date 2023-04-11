import sys
import os

from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
import pickle 
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train, y_train, x_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            #train model
            model.fit(x_train,y_train)

            #predict testing data
            y_test_pred = model.predict(x_test)

            #get r2 score for train and test data
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    except :
        pass