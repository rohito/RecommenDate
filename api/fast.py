import pandas as pd
import numpy as np
import joblib
from RecommenDate.model import Model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from RecommenDate.data import get_model, get_clean_data,get_data
from RecommenDate.similarity import similarity,get_topic,matching_topics,similarity_mean
from RecommenDate.clean_data import clean

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return dict(greeting="hello")




@app.get("/predict")
def predict(sex,orientation,essay0,essay1,essay2,essay3,essay4,essay5,essay6,essay7,essay8,essay9):
    print("beginning download")
    df_clean = get_clean_data()
    print("downloaded clean data")
    dforiginal = get_data()
    print("downloaded original dataset")
    model_list=[]
    vector_list=[]
    for i in range(10):
            vector_list.append(get_model(f"RecommenDate/models/vectorizer{i}.joblib"))
            model_list.append(get_model(f"RecommenDate/models/essay{i}.pkl"))
    print("downloaded models")
    #clean essays
    print(df_clean.head())
    d = {
        'sex':sex,
        'orientation' : orientation,
        'essay0_cleaned': essay0,
        'essay1_cleaned': essay1,
        'essay2_cleaned': essay2,
        'essay3_cleaned': essay3,
        'essay4_cleaned': essay4,
        'essay5_cleaned': essay5,
        'essay6_cleaned': essay6,
        'essay7_cleaned': essay7,
        'essay8_cleaned': essay8,
        'essay9_cleaned': essay9

    }
    df = pd.DataFrame(data=d,index=[59946])
    for i in range(10):
        df[f"essay{i}_cleaned"].fillna('',inplace=True)
        df[f"essay{i}_cleaned"]=df[f"essay{i}_cleaned"].apply(lambda x:clean(x))
        df[f"essay{i}_cleaned"]=df[f"essay{i}_cleaned"].apply(lambda x:' '.join(x))

    #build X for predict (pipeline?)
    df_clean = df_clean.append(df)

    #get vectoriser and models
    #predict
    decompose_list=[]
    components_list=[]
    for i in range(10):
        model_vectorizer = vector_list[i]
        model= model_list[i]
        model_fit=Model(df_clean[f'essay{i}_cleaned'],model_vectorizer,model)
        # data_vectorized,vocab=model_fit.vectorizer()
        # removed above line as it is happening in decompositon_svd below
        decompose,components=model_fit.decomposition_svd(n_components=500)
        decompose_list.append(decompose)
        components_list.append(components)
    df_sim=similarity_mean(decompose_list,500,index=59946)
    # res = np.argsort(df_sim)[::-1][:5]
    indexes = df_sim.index
    df_result = dforiginal.iloc[indexes[1:-1]]
    df_result = df_result.fillna('')
    if df_clean.iloc[59946]["orientation"]=="straight":
        if df_clean.iloc[59946]["sex"]=="m":
            df_result3 = df_result[(df_result["sex"]=="f") & (df_result["orientation"]=="straight")]
        if df_clean.iloc[59946]["sex"]=="f":
            df_result3 = df_result[(df_result["sex"]=="m") & (df_result["orientation"]=="straight")]
    if df_clean.iloc[59946]["orientation"]=="gay":
        if df_clean.iloc[59946]["sex"]=="m":
            df_result3 = df_result[(df_result["sex"]=="m") & (df_result["orientation"]=="gay")]
        if df_clean.iloc[59946]["sex"]=="f":
            df_result3 = df_result[(df_result["sex"]=="f") & (df_result["orientation"]=="gay")]
    if df_clean.iloc[59946]["orientation"]=="bisexual":
        df_result3 = df_result

    df_result4 = df_result3.head(3)

    topics_list=[]
    for i in range(10):
        if i == 2:
            topics_list.append(get_topic(components_list[i],200,5))
        else:
            topics_list.append(get_topic(components_list[i],500,5))

    matches =[]
    for i in range(3):
        for j in range(10):
            matches.append(matching_topics(decompose_list[j],59946,indexes[i],components_list[j])[0])

    words1=''
    words2=''
    words3=''
    for i in range(5):
        for j in range(10):
            words1 = topics_list[j][matches[j][i]]+', '+ words1
            words2 = topics_list[j][matches[j+10][i]]+', '+ words2
            words3 = topics_list[j][matches[j+20][i]]+', '+ words3

    word_list = [words1,words2,words3]
    se = pd.Series(word_list)
    df_result4["topics"] = se.values

    dict2 = df_result4.to_dict()
    return dict2
# $DELETE_END
