import pandas as pd
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
def predict(essay0,essay1,essay2,essay3,essay4,essay5,essay6,essay7,essay8,essay9):

    #get data and models
    df_clean = get_clean_data()
    # df = get_data()

    #clean essays
    d = {
        'essay0': essay0,
        'essay1': essay1,
        'essay2': essay2,
        'essay3': essay3,
        'essay4': essay4,
        'essay5': essay5,
        'essay6': essay6,
        'essay7': essay7,
        'essay8': essay8,
        'essay9': essay9

    }
    df = pd.DataFrame(data=d,index=[0])
    df["essay0"].fillna('',inplace=True)
    df["essay0"]=df["essay0"].apply(lambda x:clean(x))
    df["essay0"]=df["essay0"].apply(lambda x:' '.join(x))
    df_clean.essay0_cleaned.append(df["essay0"], ignore_index=True)

    df["essay1"].fillna('',inplace=True)
    df["essay1"]=df["essay1"].apply(lambda x:clean(x))
    df["essay1"]=df["essay1"].apply(lambda x:' '.join(x))
    df_clean.essay1_cleaned.append(df["essay1"], ignore_index=True)

    df["essay2"].fillna('',inplace=True)
    df["essay2"]=df["essay2"].apply(lambda x:clean(x))
    df["essay2"]=df["essay2"].apply(lambda x:' '.join(x))
    df_clean.essay2_cleaned.append(df["essay2"], ignore_index=True)

    df["essay3"].fillna('',inplace=True)
    df["essay3"]=df["essay3"].apply(lambda x:clean(x))
    df["essay3"]=df["essay3"].apply(lambda x:' '.join(x))
    df_clean.essay3_cleaned.append(df["essay3"], ignore_index=True)

    df["essay4"].fillna('',inplace=True)
    df["essay4"]=df["essay4"].apply(lambda x:clean(x))
    df["essay4"]=df["essay4"].apply(lambda x:' '.join(x))
    df_clean.essay4_cleaned.append(df["essay4"], ignore_index=True)

    df["essay5"].fillna('',inplace=True)
    df["essay5"]=df["essay5"].apply(lambda x:clean(x))
    df["essay5"]=df["essay5"].apply(lambda x:' '.join(x))
    df_clean.essay5_cleaned.append(df["essay5"], ignore_index=True)

    df["essay6"].fillna('',inplace=True)
    df["essay6"]=df["essay6"].apply(lambda x:clean(x))
    df["essay6"]=df["essay6"].apply(lambda x:' '.join(x))
    df_clean.essay6_cleaned.append(df["essay6"], ignore_index=True)

    df["essay7"].fillna('',inplace=True)
    df["essay7"]=df["essay7"].apply(lambda x:clean(x))
    df["essay7"]=df["essay7"].apply(lambda x:' '.join(x))
    df_clean.essay7_cleaned.append(df["essay7"], ignore_index=True)

    df["essay8"].fillna('',inplace=True)
    df["essay8"]=df["essay8"].apply(lambda x:clean(x))
    df["essay8"]=df["essay8"].apply(lambda x:' '.join(x))
    df_clean.essay8_cleaned.append(df["essay8"], ignore_index=True)

    df["essay9"].fillna('',inplace=True)
    df["essay9"]=df["essay9"].apply(lambda x:clean(x))
    df["essay9"]=df["essay9"].apply(lambda x:' '.join(x))
    df_clean.essay9_cleaned.append(df["essay9"], ignore_index=True)

    decompose_list=[]
    components_list=[]
    for i in range(10):
        model_vectorizer = get_model(f"RecommenDate/models/vectorizer{i}.joblib")
        model=get_model(f"RecommenDate/models/essay{i}.pkl")
        model_fit=Model(df_clean[f'essay{i}_cleaned'],model_vectorizer,model)
        data_vectorized,vocab=model_fit.vectorizer()
        decompose,components=model_fit.decomposition_svd(n_components=500)
        decompose_list.append(decompose)
        components_list.append(components)
    df_sim=similarity_mean(decompose_list,500,index=59946)
    df_result = df_sim.head(5)
    dict1 = {
        "essay" : df
    }
    dict2 = df_result.to_dict()
    #build X for predict (pipeline?)
    #get vectoriser and models
    #predict
    #return match

    return dict2
# $DELETE_END
