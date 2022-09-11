import os
import joblib
import pickle
import pandas as pd
from google.cloud import storage


BUCKET_NAME = "recommendate-lewagon"
BUCKET_DATA_PATH = "okcupid_profiles.csv"
BUCKET_DATACLEAN_PATH = "clean_data.csv"
# essay0model = joblib.load(open("essay0_NMF.pkl",'rb'))
# essay1model = joblib.load(open("essay1_NMF.pkl",'rb'))
# essay2model = joblib.load(open("essay2_NMF.pkl",'rb'))
# essay3model = joblib.load(open("essay3_NMF.pkl",'rb'))
# essay4model = joblib.load(open("essay4_NMF.pkl",'rb'))

def get_clean_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/data/{BUCKET_DATACLEAN_PATH}"
    df = pd.read_csv(path)
    df = df.fillna('')
    return df


def get_data( optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/data/{BUCKET_DATA_PATH}"
    df = pd.read_csv(blob)
    print(df)
    return df

def download_model( bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}'.format(
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> modle downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model

def download_vectoriser(bucket= BUCKET_NAME,rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/vectoriser/{}'.format(
        f'tfidfvectorizer0.joblib'
    )
    blob = client.blob(storage_location)
    blob.download_to_filename('vectorizer0.joblib')
    print("=> vectorisers downloaded from storage")
    model = joblib.load("vectorizer0.joblib")

    return model

def get_model(path_to_joblib):
    model = joblib.load(path_to_joblib)
    return model


if __name__ == '__main__':

    # ⚠️ in order to push a submission to kaggle you need to use the WHOLE dataset
    model = download_vectoriser()
