import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def similarity(decompose,n_components,index):
    vector=np.array(decompose.iloc[index]).reshape(1,-1)
    sim=cosine_similarity(decompose,vector).reshape(-1)
    return pd.DataFrame(sim,columns=['similarity'])


def get_topic(components,n_components,number_of_words):
    topics=[]
    for i in range(n_components):
        topic=pd.DataFrame(components.iloc[i].sort_values(ascending=False))
        topics.append(', '.join(topic.reset_index()['index'][:number_of_words]))
    return topics


def matching_topics(decompose,i,j,components):
    contribution_matrix=[]
    sum_i=[]
    sum_j=[]
    for k in range(len(components)):
        contribution_matrix.append(decompose.iloc[i][k]*decompose.iloc[j][k])
        sum_i.append(decompose.iloc[i][k]*decompose.iloc[i][k])
        sum_j.append(decompose.iloc[j][k]*decompose.iloc[j][k])
    sim=np.sum(contribution_matrix)/np.sqrt(np.sum(sum_i)*np.sum(sum_j))
    contribution_matrix=np.array(contribution_matrix/sim)
    return np.argsort(contribution_matrix)[::-1][:5],sim

def topic_matrix(topics,decompose,interest_index,svd_components):

  df=pd.DataFrame()
  for i in range(len(20)):
    match_topic=[]
    for k in range(len(matching_topics(decompose,interest_index,i,svd_components)[0])):
      match_topic.append(topics[matching_topics(decompose,interest_index,i,svd_components)[0][k]])
    df=df.append(pd.DataFrame(match_topic).T)
  return df

def similarity_mean(decompose_list,n_components,index):
    sim=[]
    for i in range(10):
        sim.append(np.abs(similarity(decompose_list[i],n_components,index)['similarity']))
    sim.append(sum(sim)/len(sim))
    df_sim=pd.DataFrame(sim).T
    df_sim.columns=['Similarity_0','Similarity_1','Similarity_2','Similarity_3','Similarity_4','Similarity_5','Similarity_6','Similarity_7','Similarity_8','Similarity_9','Similarity_mean']
    return df_sim.sort_values(by='Similarity_mean',ascending=False)*100
