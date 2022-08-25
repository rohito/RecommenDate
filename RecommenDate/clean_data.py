from contextlib import nullcontext
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import re
import string
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize

def get_data():
    data = pd.read_csv("../RecommenDate/data/okcupid_profiles.csv")
    return data

def clean_data(data):

    ###### !!!!!!!!!!!!! ######
    ###### Discuss features and remove certain stuff ######
    ###### !!!!!!!!!!!!! ######
    # For non-essay data -> no more than 50 features after encoding

    ###### Replacing null values with 'rather not say' for certain features ######
    data[['drugs', 'smokes', 'offspring', 'body_type', 'drinks']] = data[['drugs', 'smokes', 'offspring', 'body_type', 'drinks']].fillna('rather not say')

    ###### Replacing 'income' null values with mean ######
    tmp = data[data.income != -1]
    mean = tmp['income'].mean().round(1)
    data['income'] = data['income'].replace(-1, mean)

    ###### Replacing null values with logical answers(strings) for 'education', 'job' and 'speaks' features ######
    data.education=data.education.fillna('graduated from college/university').value_counts()
    data.job=data.job.fillna('other')
    data.speaks=data.speaks.fillna('english')

    ###### Cleaning 'state' feature and changing US locations to England locations ######
    data['state'] = data['location'].str.replace(r'(.*),\s','', regex=True)
    data.state.replace("california", "London", inplace=True)
    data.state.replace("new york", "Manchester", inplace=True)
    data.state.replace("illinois", "Birmingham", inplace=True)
    data.state.replace("massachusetts", "Leeds", inplace=True)
    data.state.replace("texas", "Glasgow", inplace=True)
    data.loc[(data['state'] != 'London') & (data['state'] != 'Manchester')  & (data['state'] != 'Birmingham') & (data['state'] != 'Leeds') &  (data['state'] != 'Glasgow'), "state"] = "Other"
    data['state'].value_counts()
    data.rename(columns = {'state':'city'}, inplace = True)

    ###### Cleaning the 'sign' feature ######
    imputer = SimpleImputer(strategy="most_frequent")
    imputer.fit(data[['sign']])
    data['sign'] = imputer.transform(data[['sign']])
    imputer.statistics_
    data['sign_info'] = data['sign'].apply(lambda x: get_sign(x))
    data['sign_level'] = data['sign'].apply(lambda x: get_level_of_interest(x))

    ###### Cleaning the 'religion' feature ######
    imputer2 = SimpleImputer(strategy="most_frequent") # Instantiate a SimpleImputer object with your strategy of choice
    imputer2.fit(data[['religion']]) # Call the "fit" method on the object
    data['religion'] = imputer2.transform(data[['religion']]) # Call the "transform" method on the object
    imputer2.statistics_ # The mean is stored in the transformer's memory
    data['religion_info'] = data['religion'].apply(lambda x: get_religion(x))

    ###### Replacing 'height' outliers with mean ######
    data.loc[data['height'] < 40, 'height'] = data.height.mean()

    ###### cleaning 'status' feature ######
    data['status'] = np.where(data['status'].str.contains('single'), 'available', data['status'])
    data = pd.concat([pd.get_dummies(data.status, prefix='status', prefix_sep='_'), data], axis = 1)
    data.drop("status_unknown", axis="columns", inplace=True)

    ###### replacing null values for 'diet' ######
    data.diet.fillna("no restriction", inplace=True)

    ###### create dummy variable: strict (1=strictly following a diet) ######
    data['strict'] = 0
    data.loc[data.diet.str.contains('strictly'), 'strict'] = 1
    data.loc[data.diet.str.len()==1, 'strict'] = 1

    ###### group 'diets' ######
    data['diet'] = np.where(data['diet'].str.contains('strictly anything|mostly other|anything|mostly anything|strictly other|other'), 'no restriction', data['diet'])
    data.loc[data.diet=='no restriction', 'strict'] = 0
    data['diet'] = np.where(data['diet'].str.contains('mostly vegetarian|strictly vegan|strictly vegetarian|mostly vegan|vegan|vegetarian'), 'veggie', data['diet'])
    data['diet'] = np.where(data['diet'].str.contains('mostly kosher|strictly kosher|kosher'), 'kosher', data['diet'])
    data['diet'] = np.where(data['diet'].str.contains('mostly halal|strictly halal|halal'), 'halal', data['diet'])

    ###### Cleaning 'ethnicity' feature ######
    data['eth_num'] = data.ethnicity.str.len()
    data["ethnicity2"] = "race"
    data.loc[data.eth_num<2, "ethnicity2"] = data.ethnicity.str[0]
    data.loc[data.eth_num==2, "ethnicity2"] = "biracial"
    data.loc[data.eth_num>2, "ethnicity2"] = "multiracial"
    data.loc[data.eth_num>2, "eth_num"] = 3

    ###### Cleaning 'speaks' column ######
    data['speaks_cleaned']=data.speaks.apply(lambda x:clean(x))
    data['primary_language']=data.speaks_cleaned.apply(lambda x:primary_language(x)).apply(pd.Series)[0]
    data['number_of_languages']=data.speaks_cleaned.apply(lambda x:primary_language(x)).apply(pd.Series)[1]
    data.primary_language=data.primary_language.apply(lambda x:''.join(x))
    data.number_of_languages=data.number_of_languages.apply(lambda x: x[0])

    ###### Essays ######
    data['essay0'] = data['essay0'].fillna('')
    data['essay1'] = data['essay1'].fillna('')
    data['essay2'] = data['essay2'].fillna('')
    data['essay3'] = data['essay3'].fillna('')
    data['essay4'] = data['essay4'].fillna('')
    data['essay5'] = data['essay5'].fillna('')
    data['essay6'] = data['essay6'].fillna('')
    data['essay7'] = data['essay7'].fillna('')
    data['essay8'] = data['essay8'].fillna('')
    data['essay9'] = data['essay9'].fillna('')
    data['essay0_cleaned']=data['essay0'].apply(lambda x:clean(x))
    data['essay0_cleaned']=data['essay0_cleaned'].apply(lambda x:' '.join(x))
    data['essay1_cleaned']=data['essay1'].apply(lambda x:clean(x))
    data['essay1_cleaned']=data['essay1_cleaned'].apply(lambda x:' '.join(x))
    data['essay2_cleaned']=data['essay2'].apply(lambda x:clean(x))
    data['essay2_cleaned']=data['essay2_cleaned'].apply(lambda x:' '.join(x))
    data['essay3_cleaned']=data['essay3'].apply(lambda x:clean(x))
    data['essay3_cleaned']=data['essay3_cleaned'].apply(lambda x:' '.join(x))
    data['essay4_cleaned']=data['essay4'].apply(lambda x:clean(x))
    data['essay4_cleaned']=data['essay4_cleaned'].apply(lambda x:' '.join(x))
    data['essay5_cleaned']=data['essay5'].apply(lambda x:clean(x))
    data['essay5_cleaned']=data['essay5_cleaned'].apply(lambda x:' '.join(x))
    data['essay6_cleaned']=data['essay6'].apply(lambda x:clean(x))
    data['essay6_cleaned']=data['essay6_cleaned'].apply(lambda x:' '.join(x))
    data['essay7_cleaned']=data['essay7'].apply(lambda x:clean(x))
    data['essay7_cleaned']=data['essay7_cleaned'].apply(lambda x:' '.join(x))
    data['essay8_cleaned']=data['essay8'].apply(lambda x:clean(x))
    data['essay8_cleaned']=data['essay8_cleaned'].apply(lambda x:' '.join(x))
    data['essay9_cleaned']=data['essay9'].apply(lambda x:clean(x))
    data['essay9_cleaned']=data['essay9_cleaned'].apply(lambda x:' '.join(x))
    return clean_data

def primary_language(text):
        primary_lang=[]
        number_of_languages=[]
        counter=0
        if 'fluently' in text:
            primary_lang.append(text[text.index('fluently')-1])
            counter=counter+1
        else:
            primary_lang.append(text[0])

        if 'poorly' in text:
            counter=counter+1
        if 'okay' in text:
            counter=counter+1
        number_of_languages.append(len(text)-counter)
        return primary_lang,number_of_languages

def get_sign(text):
    return text.split(' ')[0]

def get_level_of_interest(text):
    x = text.split(' ')[2:]
    return ' '.join(x)

def get_religion(text):
    return text.split(' ')[0]

    ###### Essays ######
def clean (text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ') # Remove Punctuation
    lowercased = text.lower() # Lower Case
    tokenized = word_tokenize(lowercased) # Tokenize
    words_only = [word for word in tokenized if word.isalpha()] # Remove numbers
    stop_words = set(stopwords.words('english')) # Make stopword list
    without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
    lemma=WordNetLemmatizer() # Initiate Lemmatizer
    lemmatized = [lemma.lemmatize(word) for word in without_stopwords] # Lemmatize
    return lemmatized




    ####################################################################

    ###### Cleaning 'religion' feature ######
    # (data.religion.isnull().sum()/len(data))*100
    # data["religion"] = data["religion"].fillna('agnostic')
    # pd.set_option('display.max_columns', None)

    # dfagn = data.copy()
    # dfagn["religion"] = data["religion"].fillna("agnosticism")
    # dfother = data.copy()
    # dfother["religion"] = data["religion"].fillna("other")
    # dfagn['rel_scale'] = np.where(dfagn['religion'].str.contains("very serious"), 1, 0)

    # def create_religion_scale(data):
    #     conditions = [
    #         data['religion'].str.contains("very serious"),
    #         data['religion'].str.contains("somewhat serious"),
    #         data['religion'].str.contains("not too serious"),
    #         data['religion'].str.contains("laughing")]
    #     choices = ['very serious', 'somewhat serious', 'not too serious','laughing']
    #     data['rel_scale'] = np.select(conditions, choices, default='normal')
    #     return data

    ###### Should I replace this with 'rather not say' ###### searchterm
    # data.religion.fillna("other", inplace=True)

    # dfagn = create_religion_scale(dfagn)
    # dfother = create_religion_scale(dfother)

    ###### create 1 variable: serious (1=yes, 0=neutral, -1=no) ######
    # data["serious"] = 0
    # data.loc[data.religion.str.contains("very|somewhat"), "serious"] = 1
    # data.loc[data.religion.str.contains("laughing"), "serious"] = -1
    # data.religion = data.religion.str.split().str[0]

    ####################################################################
