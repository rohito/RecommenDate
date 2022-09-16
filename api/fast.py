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
def predict(essay0,essay1,essay2,essay3,essay4,essay5,essay6,essay7,essay8,essay9):
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
    indexes = df_sim.head(4).index
    df_result = dforiginal.iloc[indexes[1:]]
    df_result = df_result.fillna('')
    dict2 = df_result.to_dict()

    #return match

    return dict2
# $DELETE_END
