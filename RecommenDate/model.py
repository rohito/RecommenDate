import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pickle
import os
from math import sqrt
from RecommenDate.data import get_model, get_clean_data
from RecommenDate.similarity import similarity,get_topic,matching_topics,similarity_mean
# RecommenDate/vectorizer0.joblib
# from sklearn.pipeline import make_pipeline


class Model():
  def __init__(self,X,model_vectorizer,model):
    self.X=X
    self.model_vectorizer=model_vectorizer
    self.model=model


  def vectorizer(self):
    data_vectorized = self.model_vectorizer.transform(self.X)
    vocab = self.model_vectorizer.get_feature_names_out()
    return data_vectorized,vocab

  def decomposition_svd(self,n_components):
    self.data_vectorized,self.vocab=Model.vectorizer(self)
    decompose=pd.DataFrame(self.model.transform(self.data_vectorized),index=self.X)
    components=self.model.components_
    svd_components=pd.DataFrame(components,columns=self.vocab)
    # svd_explained=self.model.explained_variance_ratio_
    return decompose,svd_components #,svd_explained

  def decomposition_NMF(self,model,n_components,essay):
    decompose_NMF = pd.DataFrame(model.transform(self.data_vectorized),index=self.X)
    components = model.components_
    NMF_components = pd.DataFrame(components,columns=vocab)
    return decompose_NMF,NMF_components



if __name__ == '__main__':
  data = get_clean_data()
  decompose_list=[]
  components_list=[]
  for i in range(10):
    model_vectorizer = get_model(f"models/vectorizer{i}.joblib")
    model=get_model(f"models/essay{i}.pkl")
    model_fit=Model(data[f'essay{i}_cleaned'],model_vectorizer,model)
    data_vectorized,vocab=model_fit.vectorizer()
    decompose,components,svd_explained=model_fit.decomposition_svd(n_components=500)
    decompose_list.append(decompose)
    components_list.append(components)
  df_sim=similarity_mean(decompose_list,500,index=3)
  print(df_sim)
