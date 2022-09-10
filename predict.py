import os
from math import sqrt

import joblib
import pandas as pd
from google.cloud import storage


BUCKET_NAME = "recommendate-lewagon"
BUCKET_DATA_PATH = "recommendate-lewagon/okcupid_profiles.csv"
BUCKET_DATACLEAN_PATH = "recommendate-lewagon/clean_data.csv"

def get_clean_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/{BUCKET_DATACLEAN_PATH}"
    df = pd.read_csv(path)
    print(df)
    return df


def get_data( optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/{BUCKET_DATA_PATH}"
    df = pd.read_csv(path)
    print(df)
    return df


def download_model( bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}'.format(
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model

def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline


if __name__ == '__main__':

    # ⚠️ in order to push a submission to kaggle you need to use the WHOLE dataset
    cleandf = get_data_from_gcp()
    cleandf.head()
