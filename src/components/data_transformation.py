import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
import os

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.utils import save_object 


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            #Defining which columns should be scaled and which to be ordinal encoded
            categorical_cols = ['cut','color','clarity']
            numerical_cols = ['carat','depth','table','x','y','z']

            #Defining ranking for each ordinal values for categorical columns
            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Pipeline initiated")

            ##Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )

            ##Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor
        
            logging.info("Pipeline completed")

        except Exception as e:
            logging.info("Error in data transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            #Reading train and test data 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")
            logging.info(f'Train DataFrame Head : \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head : \n{test_df.head().to_string()}')

            logging.info("Obtaining preprocessing object ")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing database")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #np.c_ = concatenation of 2 arrays, we will be able to apply ML algorithms v easily
            #also numpy arrays are super fast
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            #calling func from utils ,to save pickle file
            save_object (
                
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("Preprocessor pickle file saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occured in the initiate data transformation")

            raise CustomException(e,sys)