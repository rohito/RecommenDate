import os
from math import sqrt

import joblib
import pandas as pd
from google.cloud import storage


BUCKET_NAME = "recommendate-lewagon"
BUCKET_DATA_PATH = "okcupid_profiles.csv"
BUCKET_DATACLEAN_PATH = "clean_data.csv"

def get_clean_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/data/{BUCKET_DATACLEAN_PATH}"
    df = pd.read_csv(path)
    print(df)
    return df


def get_data( optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    bucket = client.get_bucket("recommendate-lewagon")
    path = f"/data/{BUCKET_DATA_PATH}"
    blob = bucket.get_blob(path)
    df = pd.read_csv(blob)
    print(df)
    return df


if __name__ == '__main__':

    # ⚠️ in order to push a submission to kaggle you need to use the WHOLE dataset
    cleandf = get_data()
    cleandf.head()
