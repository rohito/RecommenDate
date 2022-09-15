import os
import joblib
import pickle
import pandas as pd
from google.cloud import storage
# from io import BytesIO
# from tensorflow.python.lib.io import file_io


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
    df = pd.read_csv(path)
    print(df)
    return df

def download_model( bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/pickle_files_svd/{}'.format(
        'essay3.pkl')
    blob = client.blob(storage_location)
    blob.download_to_filename('svd3.pkl')

    # path_essay1 = f"gs://{BUCKET_NAME}/models/pickle_files_svd/essay1.pkl"
    # f = BytesIO(file_io.read_file_to_string(path_essay1,binary_mode=True))
    # model = pickle.load()
    print("=> modle downloaded from storage")
    # with open("svd.pkl", 'rb') as pickle_file:
    #     content = pickle.load(pickle_file)
    # model = content
    # return model
    model = joblib.load("svd3.pkl")
    return model


def download_vectoriser(bucket= BUCKET_NAME,rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/vectoriser/{}'.format(
        f'tfidfvectorizer9.joblib'
    )
    blob = client.blob(storage_location)
    blob.download_to_filename('vectorizer9.joblib')
    print("=> vectorisers downloaded from storage")
    model = joblib.load("vectorizer9.joblib")

    return model

def get_model(path_to_joblib):
    model = joblib.load(path_to_joblib)
    return model


if __name__ == '__main__':

    # ⚠️ in order to push a submission to kaggle you need to use the WHOLE dataset
    df = get_data()
    print(df.head())
