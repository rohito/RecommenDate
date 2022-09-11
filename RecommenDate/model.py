import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pickle
import os
from math import sqrt
from data import get_model, get_clean_data
# RecommenDate/vectorizer0.joblib


model = get_model("vectorizer0.joblib")
def vectorizer(essay,vectorizer=model):
  data_vectorized = vectorizer.transform(essay)
  vocab = vectorizer.get_feature_names()
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





if __name__ == '__main__':


    df = get_clean_data()
    model = get_model("vectorizer0.joblib")
    es0,vocab = vectorizer(df.essay0_cleaned,model)
    print(es0[0])
