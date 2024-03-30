import os
import sys
import pandas as pd
import pickle
from keras.models import load_model, save_model
from src.exception.exception import CustomException


def save_csv(file_path, data):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        data.to_csv(file_path, index=False)

    except Exception as e:
        raise CustomException(e)


def load_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        return data

    except Exception as e:
        raise CustomException(e)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def save_GRU(file_path, model):
    try:
        model.save(file_path)
        print(f"Model saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_GRU(file_path):
    try:
        model = load_model(file_path)
        print(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        raise CustomException(e, sys)