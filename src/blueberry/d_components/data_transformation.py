import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from blueberry.a_constants import *
from blueberry import logging, CustomException
from blueberry.b_entity.config_entity import DataTransformationConfig
from blueberry.f_utils.common import read_yaml


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    # Note: You can add different data transformation techniques such as Scaler, PCA and all
    # You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up

    def train_test_spliting(self):
        try:
            RANDOM_STATE = read_yaml(
                GLOBALPARAMS_FILE_PATH).globalparams.RANDOM_STATE
            df_syn = pd.read_csv(self.config.data_path_syn)
            df_syn.drop('id', axis=1, inplace=True)
            df_act = pd.read_csv(self.config.data_path_act)
            df_act.drop('Row#', axis=1, inplace=True)

            # Merging both datasets
            df = pd.concat([df_syn, df_act], axis=0)
            df.reset_index(drop=True, inplace=True)

            # Dropping the duplicates & restting index
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Split the data into training and test sets. (0.75, 0.25) split.
            train, test = train_test_split(
                df, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)

            train.to_csv(os.path.join(
                self.config.root_dir, "train.csv"), index=False)
            test.to_csv(os.path.join(
                self.config.root_dir, "test.csv"), index=False)

            logging.info("Splited data into training and test sets")
            logging.info(
                f" Train Shape ==> {train.shape} | test Shape ==> {test.shape}")
        except Exception as e:
            logging.error(CustomException(e, sys))
            raise CustomException(e, sys)
