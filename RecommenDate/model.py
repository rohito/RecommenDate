import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pickle
import os
from math import sqrt
from google.cloud import storage


BUCKET_NAME = "recommendate-lewagon"
BUCKET_DATA_PATH = "recommendate-lewagon/okcupid_profiles.csv"
BUCKET_DATACLEAN_PATH = "recommendate-lewagon/clean_data.csv"
# essay0model = joblib.load(open("essay0_NMF.pkl",'rb'))
# essay1model = joblib.load(open("essay1_NMF.pkl",'rb'))
# essay2model = joblib.load(open("essay2_NMF.pkl",'rb'))
# essay3model = joblib.load(open("essay3_NMF.pkl",'rb'))
# essay4model = joblib.load(open("essay4_NMF.pkl",'rb'))


def vectorizer(essay):
  data_vectorized = vectorizer.transform(essay)
  joblib.dump(vectorizer,'tfidfvectorizer.joblib')
  vocab = vectorizer.get_feature_names_out()
  return data_vectorized,vocab


def similarity(decompose,index):
  vector=np.array(decompose.iloc[index]).reshape(1,-1)
  sim=cosine_similarity(decompose,vector).reshape(-1)
  return pd.DataFrame(sim,columns=['similarity']).sort_values(by='similarity',ascending=False)


def get_topic(components,n_components,number_of_words):
  topics=[]
  for i in range(n_components):
    X=pd.DataFrame(components.iloc[i].sort_values(ascending=False))
    topics.append(', '.join(X.reset_index()['index'][:number_of_words]))
  return topics


def matching_topics(decompose,i,j,components):
  contribution_matrix=[]
  sum_i=[]
  sum_j=[]
  for k in range(len(components)):
    contribution_matrix.append(decompose.iloc[i][k]*decompose.iloc[j][k])
    sum_i.append(decompose.iloc[i][k]*decompose.iloc[i][k])
    sum_j.append(decompose.iloc[j][k]*decompose.iloc[j][k])
  sim2=cosine_similarity(np.expand_dims(np.array(decompose.iloc[i]),0),np.expand_dims(np.array(decompose.iloc[j]),0))
  sim=np.sum(contribution_matrix)/np.sqrt(np.sum(sum_i)*np.sum(sum_j))
  contribution_matrix=np.array(contribution_matrix/sim)
  return np.argsort(contribution_matrix)[::-1][:5],sim

def similarity_matrix(topics,decompose,interest_index,pca_components):

  df=pd.DataFrame()
  for i in range(len(decompose)):
    match_topic=[]
    for k in range(len(matching_topics(decompose,interest_index,i,pca_components)[0])):
      match_topic.append(topics[matching_topics(decompose,interest_index,i,pca_components)[0][k]])
    df=df.append(pd.DataFrame(match_topic).T)
  return df

def decomposition_NMF(model,n_components,essay):
  vectorized_data,vocab = vectorizer(essay)
  decompose_NMF = pd.DataFrame(model.transform(vectorized_data),index=essay)
  components = model.components_
  NMF_components = pd.DataFrame(components,columns=vocab)
  return decompose_NMF,NMF_components



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
